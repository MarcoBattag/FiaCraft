from argparse import ArgumentParser
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch as th
import numpy as np
from collections import deque
import sys
import gym
from gym.envs.registration import register

# --- 1. SETUP PERCORSI E IMPORT CUSTOM ---
# Aggiungiamo la cartella corrente al path per trovare 'envs'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# TENTATIVO DI IMPORTARE L'AMBIENTE CUSTOM
try:
    # Importiamo direttamente la funzione entrypoint dal file FIAenv.py
    from envs.FIAenv import fia_pickaxe_entrypoint
    print("[*] Modulo 'envs.FIAenv' importato correttamente.")
except ImportError as e:
    print(f"\n[ERRORE CRITICO] Impossibile importare 'envs.FIAenv': {e}")
    print("Assicurati che la cartella 'envs' e il file 'FIAenv.py' siano nella stessa directory di questo script.")
    sys.exit(1)

# REGISTRAZIONE MANUALE DELL'AMBIENTE (Per evitare NameNotFound)
if 'FIA-WoodenPickaxe-v0' in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs['FIA-WoodenPickaxe-v0']

register(
    id='FIA-WoodenPickaxe-v0',
    entry_point=fia_pickaxe_entrypoint,
    max_episode_steps=6000
)
print("[*] Ambiente 'FIA-WoodenPickaxe-v0' registrato manualmente.")

# --- IMPORT MINERL E VPT ---
import minerl
try:
    from openai_vpt.agent import MineRLAgent
    from openai_vpt.lib.tree_util import tree_map
except ImportError:
    print("[!] Libreria openai_vpt non trovata nel path standard. Provo path relativo..")
    sys.path.append(os.path.abspath('../'))
    from openai_vpt.agent import MineRLAgent
    from openai_vpt.lib.tree_util import tree_map

# --- CONFIGURAZIONE GPU ---
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"[*] Dispositivo attivo: {device}")
if device.type == 'cuda':
    print(f"[*] GPU: {th.cuda.get_device_name(0)}")

# --- CONFIGURAZIONE REWARDS ---
MATERIAL_REWARDS = {
    "log": 1.0, 
    "planks": 2.0,
    "stick": 5.0,
    "crafting_table": 15.0, 
    "wooden_pickaxe": 100.0,
    "dirt": -0.05,
}

# --- FUNZIONI DI SUPPORTO ---
def compute_reward_and_update_best(inventory, best_inventory):
    reward = 0
    for material, value in MATERIAL_REWARDS.items():
        # Gestione robusta per inventory (dict o array)
        current_val = inventory.get(material, 0)
        if isinstance(current_val, np.ndarray):
            current_quantity = int(current_val.item()) if current_val.size == 1 else int(np.max(current_val))
        else:
            current_quantity = int(current_val)

        max_ever_quantity = int(best_inventory.get(material, 0))

        if current_quantity > max_ever_quantity:
            diff = current_quantity - max_ever_quantity
            if value > 0.5:
                print(f" > PROGRESSO: {material} +{diff} (Tot: {current_quantity}). Reward: +{diff * value}")
            reward += diff * value
            best_inventory[material] = current_quantity
    return reward

def action_based_reward(action, jump_window):
    reward = 0
    if sum(jump_window) > 20: 
        reward -= 0.02
    
    # Check if active (handle torch tensors or numpy arrays)
    is_active = False
    for v in action.values():
        val = v.item() if isinstance(v, th.Tensor) else v
        if isinstance(val, (int, float)) and val == 1:
            is_active = True
            break
        elif isinstance(val, np.ndarray) and np.any(val == 1):
            is_active = True
            break
            
    if not is_active:
        reward -= 0.01
    return reward

def normalize(tensor):
    if tensor.numel() <= 1: return tensor
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

# --- MAIN LOOP ---
def main(model_path, weights_path, env_name, n_episodes=20, max_steps=6000, show=True):
    # 1. Creazione Ambiente (Usa gym standard, non aicrowd)
    print(f"[*] Avvio ambiente: {env_name}...")
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"[ERRORE FATALE] Gym non riesce a creare l'ambiente: {e}")
        sys.exit(1)
    
    # 2. Caricamento Modello
    print(f"[*] Caricamento modello da: {model_path}")
    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    
    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_path)
    agent.policy.to(device) # Sposta policy su GPU
    print("[*] Modello caricato su GPU.")

    # Freeze pesi base
    for param in agent.policy.parameters(): param.requires_grad = False
    for param in agent.policy.pi_head.parameters(): param.requires_grad = True

    optimizer = th.optim.RMSprop(
        filter(lambda p: p.requires_grad, agent.policy.parameters()), 
        lr=0.00001, alpha=0.99, eps=1e-8
    )
    
    # Cartelle output
    base_dir = "./data/Stats/RL_Corrected"
    os.makedirs(base_dir, exist_ok=True)
    cumulative_rewards_all = []

    # --- INIZIO TRAINING ---
    for episode in range(n_episodes):
        print(f"\n=== AVVIO EPISODIO {episode + 1}/{n_episodes} ===")
        obs = env.reset()
        
        # Inizializza stato ricorrente su GPU
        agent_state = agent.policy.initial_state(1)
        agent_state = tree_map(lambda x: x.to(device) if x is not None else None, agent_state)

        best_inventory = {key: 0 for key in MATERIAL_REWARDS.keys()}
        jump_window = deque(maxlen=20)
        cumulative_episode_reward = 0
        
        episode_rewards = []
        steps_plot = []
        batch_log_probs, batch_advantages = [], []

        for step in range(max_steps):
            # 1. Action Step
            # Nota: get_action restituisce (action, state, ...)
            action, agent_state, _ = agent.get_action(obs, state=agent_state)
            
            if "ESC" in action: action["ESC"] = 0
            
            next_obs, _, done, _ = env.step(action)
            
            # 2. Reward Calculation
            inventory = next_obs["inventory"]
            jump_window.append(action.get("jump", 0))
            
            rew_mat = compute_reward_and_update_best(inventory, best_inventory)
            rew_act = action_based_reward(action, jump_window)
            reward = rew_mat + rew_act
            cumulative_episode_reward += reward

            if inventory.get("wooden_pickaxe", 0) > 0:
                print("\n!!! OBIETTIVO RAGGIUNTO !!!")
                reward += 500
                cumulative_episode_reward += 500
                done = True

            # 3. PPO Preparation
            # Converti osservazione e azione correnti in tensori GPU
            agent_obs = tree_map(lambda x: x.to(device) if isinstance(x, th.Tensor) else x, agent._env_obs_to_agent(obs))
            ac_torch = tree_map(lambda x: x.to(device) if isinstance(x, th.Tensor) else x, agent._env_action_to_agent(action, to_torch=True, check_if_null=False))
            
            # Ottieni distribuzione probabilità
            pi_dist, _, _ = agent.policy.get_output_for_observation(
                agent_obs, agent_state, th.tensor([False]).to(device)
            )
            log_prob = agent.policy.get_logprob_of_action(pi_dist, ac_torch)
            
            # Advantage (semplificato 1-step per stabilità)
            adv = reward + 0.99 * log_prob.detach().mean() - log_prob.mean()

            batch_log_probs.append(log_prob)
            batch_advantages.append(adv)

            # 4. Update Weights
            if (step + 1) % 64 == 0 or done:
                if len(batch_log_probs) > 0:
                    b_log_prob = th.stack(batch_log_probs).to(device)
                    b_adv = th.stack(batch_advantages).to(device)
                    b_adv = normalize(b_adv)

                    # PPO Clip
                    ratio = th.exp(b_log_prob - b_log_prob.detach())
                    surr1 = ratio * b_adv
                    surr2 = th.clamp(ratio, 0.8, 1.2) * b_adv
                    loss = -th.min(surr1, surr2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
                    optimizer.step()
                    
                    if step % 64 == 0:
                        print(f"\rStep: {step} | Reward: {cumulative_episode_reward:.2f} | Loss: {loss.item():.4f}", end="")
                    
                    batch_log_probs, batch_advantages = [], []

            obs = next_obs
            if show: env.render()
            if done: break

        cumulative_rewards_all.append(cumulative_episode_reward)
        print(f"\nFine Episodio {episode+1}. Reward Totale: {cumulative_episode_reward:.2f}")
        
        # Salvataggio intermedio
        if cumulative_episode_reward > 50:
             th.save(agent.policy.state_dict(), f"{base_dir}/backup_ep{episode}.weights")

    env.close()
    
    # Save Final
    th.save(agent.policy.state_dict(), "ppo_wooden_pickaxe_final.weights")
    print("Addestramento completato.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default="FIA-WoodenPickaxe-v0")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    main(args.model, args.weights, args.env, args.episodes, args.max_steps, args.show)