from argparse import ArgumentParser
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import aicrowd_gym
import minerl
import torch as th
import numpy as np
from collections import deque
import sys

# --- SETUP PERCORSI E IMPORT ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from openai_vpt.agent import MineRLAgent
    # Importante per spostare dizionari complessi sulla GPU
    from openai_vpt.lib.tree_util import tree_map 
except ImportError:
    sys.path.append(os.path.abspath('../'))
    from openai_vpt.agent import MineRLAgent
    from openai_vpt.lib.tree_util import tree_map

# --- CONFIGURAZIONE DISPOSITIVO (GPU/CPU) ---
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"[*] Dispositivo di addestramento: {device}")
if device.type == 'cuda':
    print(f"[*] GPU: {th.cuda.get_device_name(0)}")

# --- CONFIGURAZIONE REWARDS ---
MATERIAL_REWARDS = {
    "log": 1.0, 
    "planks": 2.0,
    "stick": 5.0,
    "crafting_table": 15.0, 
    "wooden_pickaxe": 100.0,
    # Penalità lievi
    "dirt": -0.05,
}

JUMP_THRESHOLD = 10
JUMP_WINDOW = 40 

# --- FUNZIONI DI SUPPORTO ---

def compute_reward_and_update_best(inventory, best_inventory):
    """Calcola reward basata sull'acquisizione di NUOVI oggetti (High Water Mark)."""
    reward = 0
    for material, value in MATERIAL_REWARDS.items():
        # Gestione robusta per inventory che potrebbe essere un dizionario o array
        current_quantity = int(inventory.get(material, 0)) if isinstance(inventory, dict) else 0
        max_ever_quantity = int(best_inventory.get(material, 0))

        if current_quantity > max_ever_quantity:
            diff = current_quantity - max_ever_quantity
            # Stampa solo per progressi significativi
            if value > 0.5:
                print(f" > PROGRESSO: {material} +{diff} (Tot: {current_quantity}). Reward: +{diff * value}")
            reward += diff * value
            best_inventory[material] = current_quantity
    return reward

def action_based_reward(action, jump_window):
    """Penalità per comportamenti indesiderati."""
    reward = 0
    # Penalità salti eccessivi
    if sum(jump_window) > 20: 
        reward -= 0.02
    # Penalità inattività
    is_active = any(np.any(v == 1) if isinstance(v, np.ndarray) else v == 1 for v in action.values())
    if not is_active:
        reward -= 0.01
    return reward

def normalize(tensor):
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

# --- MAIN LOOP ---

def main(model_path, weights_path, env_name, n_episodes=10, max_steps=2000, show=True):
    env = aicrowd_gym.make(env_name)
    
    # Caricamento parametri
    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    
    # Inizializzazione Agente con supporto GPU
    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_path)
    print("[*] Modello caricato e spostato su GPU.")

    # Spostiamo la policy su GPU esplicitamente
    agent.policy.to(device)

    # Setup Grafico
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Reward Cumulativa (Live)")
    ax.set_xlabel("Passi")
    ax.set_ylabel("Reward")
    line, = ax.plot([], [], label="Reward Episodio")
    plt.legend()
    plt.grid(True)

    # Freeze dei parametri (Tranne la testa della policy)
    for param in agent.policy.parameters():
        param.requires_grad = False
    for param in agent.policy.pi_head.parameters():
        param.requires_grad = True

    optimizer = th.optim.RMSprop(
        filter(lambda p: p.requires_grad, agent.policy.parameters()), 
        lr=0.00001, alpha=0.99, eps=1e-8
    )
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    gamma = 0.99 

    base_dir = "./data/Stats/RL_GPU_Training"
    os.makedirs(base_dir, exist_ok=True)
    cumulative_rewards_all = []

    for episode in range(n_episodes):
        print(f"\n=== AVVIO EPISODIO {episode + 1}/{n_episodes} ===")
        obs = env.reset()
        
        # Stato nascosto per la rete ricorrente (LSTM)
        # Importante: Inizializzarlo e spostarlo su GPU
        agent_state = agent.policy.initial_state(1)
        agent_state = tree_map(lambda x: x.to(device) if x is not None else None, agent_state)

        best_inventory = {key: 0 for key in MATERIAL_REWARDS.keys()}
        jump_window = deque(maxlen=20)
        cumulative_episode_reward = 0
        
        episode_rewards = []
        steps_plot = []
        
        batch_rewards = []
        batch_log_probs = []
        batch_advantages = []

        episode_dir = f"{base_dir}/episode{episode + 1}"
        os.makedirs(episode_dir, exist_ok=True)

        for step in range(max_steps):
            # 1. Scelta Azione (Inferenza)
            # L'agente gestisce internamente la conversione obs -> gpu per l'inferenza base
            action, agent_state, _ = agent.get_action(obs, state=agent_state) # Passiamo lo stato ricorrente
            
            # Action manipulation
            if "ESC" in action: action["ESC"] = 0
            
            # 2. Esecuzione Step
            next_obs, _, done, _ = env.step(action)
            
            # 3. Calcolo Reward
            inventory = next_obs["inventory"]
            jump_window.append(action.get("jump", 0))
            
            material_reward = compute_reward_and_update_best(inventory, best_inventory)
            act_reward = action_based_reward(action, jump_window)
            reward = material_reward + act_reward
            cumulative_episode_reward += reward

            # Win condition
            if inventory.get("wooden_pickaxe", 0) > 0:
                print("\n!!! OBIETTIVO RAGGIUNTO: PICCONE DI LEGNO !!!")
                reward += 500
                cumulative_episode_reward += 500
                done = True

            # 4. Preparazione PPO (GPU HANDLING CRITICO)
            # Convertiamo l'osservazione corrente in tensori GPU
            agent_obs = agent._env_obs_to_agent(obs)
            agent_obs = tree_map(lambda x: x.to(device) if isinstance(x, th.Tensor) else x, agent_obs)
            
            # Calcoliamo la distribuzione di probabilità dell'azione presa
            # Nota: Usiamo lo stato "vecchio" (prima dell'update) per calcolare la log_prob
            # Per semplificare, in VPT spesso si usa lo stato corrente o si ignora in short-horizon,
            # ma qui usiamo agent_state (che è stato aggiornato da get_action). 
            # Per rigore matematico servirebbe lo stato 'pre-azione', ma per fine-tuning leggero questo è accettabile.
            
            pi_distribution, _, _ = agent.policy.get_output_for_observation(
                agent_obs,
                agent_state, # Qui usiamo lo stato ricorrente corrente
                th.tensor([False]).to(device) # dummy first
            )
            
            # Convertiamo l'azione presa in tensore GPU
            ac_torch = agent._env_action_to_agent(action, to_torch=True, check_if_null=False)
            ac_torch = tree_map(lambda x: x.to(device) if isinstance(x, th.Tensor) else x, ac_torch)

            log_prob = agent.policy.get_logprob_of_action(pi_distribution, ac_torch)

            # Calcolo Advantage (Semplificato)
            # Normalizziamo su GPU
            advantage = reward + gamma * log_prob.detach().mean() - log_prob.mean()
            
            # Appendiamo ai batch (mantenendo il grafo computazionale per log_prob)
            batch_rewards.append(reward)
            batch_log_probs.append(log_prob)
            batch_advantages.append(advantage)

            # 5. PPO Update Step
            if (step + 1) % 64 == 0 or done:
                if len(batch_log_probs) > 0:
                    # Stack su GPU
                    batch_log_probs_t = th.stack(batch_log_probs).to(device)
                    batch_advantages_t = th.stack(batch_advantages).to(device)
                    batch_advantages_t = normalize(batch_advantages_t) # Normalizzazione finale

                    # PPO Clipping Logic
                    r_t = th.exp(batch_log_probs_t - batch_log_probs_t.detach())
                    epsilon = 0.2
                    clipped_ratio = th.clamp(r_t, 1 - epsilon, 1 + epsilon)
                    
                    # Loss calculation
                    loss = -th.min(r_t * batch_advantages_t, clipped_ratio * batch_advantages_t).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    if step % 64 == 0:
                        print(f"\rStep: {step} | Reward: {cumulative_episode_reward:.2f} | Loss: {loss.item():.4f}", end="")
                    
                    # Reset Batch
                    batch_rewards = []
                    batch_log_probs = []
                    batch_advantages = []

            # Aggiornamento Grafico (ogni 20 step per non rallentare)
            if step % 20 == 0:
                steps_plot.append(step + 1)
                episode_rewards.append(cumulative_episode_reward)
                line.set_xdata(steps_plot)
                line.set_ydata(episode_rewards)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
                plt.pause(0.001)

            obs = next_obs # Update observation for next loop
            if show:
                env.render()
            if done:
                break

        # --- FINE EPISODIO ---
        cumulative_rewards_all.append(cumulative_episode_reward)
        df = pd.DataFrame({"Passo": steps_plot, "Reward Cumulativa": episode_rewards})
        df.to_excel(os.path.join(episode_dir, f"episode_{episode + 1}_rewards.xlsx"), index=False)
        
        # Save checkpoint se buono
        if cumulative_episode_reward > 50:
             th.save(agent.policy.state_dict(), f"checkpoints/backup_ep{episode}.weights")

    env.close()

    # Salvataggio Finale
    plt.figure()
    plt.plot(range(1, n_episodes + 1), cumulative_rewards_all, marker='o')
    plt.title("Trend Addestramento GPU")
    plt.savefig(f"{base_dir}/total_trend.png")
    
    th.save(agent.policy.state_dict(), "ppo_wooden_pickaxe_gpu.weights")
    print("\nAddestramento completato.")

if __name__ == "__main__":
    parser = ArgumentParser("PPO MineRL GPU")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--model", type=str, required=True, help=".model path")
    parser.add_argument("--weights", type=str, required=True, help=".weights path")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, args.episodes, args.max_steps, args.show)