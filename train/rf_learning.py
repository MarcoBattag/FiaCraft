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
import envs

# Setup percorsi (mantenuto dal tuo script originale)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from openai_vpt.agent import MineRLAgent # Decommenta se necessario in base alla tua struttura
# Assumo che MineRLAgent sia importabile dato il tuo script precedente
try:
    from openai_vpt.agent import MineRLAgent
except ImportError:
    # Fallback se la struttura delle cartelle è diversa
    sys.path.append(os.path.abspath('../'))
    from openai_vpt.agent import MineRLAgent

# --- CONFIGURAZIONE REWARDS ---
# La logica è: Materiali Grezzi < Materiali Lavorati < Utensili < OBIETTIVO
MATERIAL_REWARDS = {
    # Materiali Grezzi (Reward bassa per incentivare la raccolta, ma non il farming inutile)
    "log": 1.0, 
    "birch_log": 1.0,
    "dark_oak_log": 1.0,
    "jungle_log": 1.0,
    "oak_log": 1.0,
    "spruce_log": 1.0,
    "acacia_log": 1.0,
    
    # Materiali Intermedi (Necessari per il crafting)
    "planks": 2.0,
    "birch_planks": 2.0,
    "dark_oak_planks": 2.0,
    "jungle_planks": 2.0,
    "oak_planks": 2.0,
    "spruce_planks": 2.0,
    "acacia_planks": 2.0,
    
    # STEP FONDAMENTALE MANCANTE PRIMA: Bastoncini
    "stick": 5.0,
    
    # Crafting Station
    "crafting_table": 15.0, 
    
    # OBIETTIVO FINALE (Reward molto alta)
    "wooden_pickaxe": 100.0,

    # Penalità lievi per spazzatura (opzionale)
    "dirt": -0.05,
    "gravel": -0.05,
    "sand": -0.05
}

# Configurazione per monitorare i salti
JUMP_THRESHOLD = 10
JUMP_WINDOW = 40 

# --- FUNZIONI DI SUPPORTO ---

def compute_reward_and_update_best(inventory, best_inventory):
    """
    Calcola la reward basata sull'acquisizione di NUOVI oggetti rispetto
    al massimo storico posseduto (High Water Mark).
    """
    reward = 0
    # Controlliamo solo gli oggetti che ci interessano nel dizionario REWARDS
    for material, value in MATERIAL_REWARDS.items():
        current_quantity = int(inventory.get(material, 0))
        max_ever_quantity = int(best_inventory.get(material, 0))

        # Diamo reward SOLO se l'inventario corrente supera il record precedente.
        # Questo premia l'ottenimento, ma non penalizza il consumo (crafting).
        if current_quantity > max_ever_quantity:
            diff = current_quantity - max_ever_quantity
            print(f">>> PROGRESSO: {material} aumentato a {current_quantity} (Nuovo Record). Reward: +{diff * value}")
            reward += diff * value
            
            # Aggiorniamo il record storico per questo materiale
            best_inventory[material] = current_quantity

    return reward

def action_based_reward(action, jump_window):
    """
    Penalità per comportamenti indesiderati (troppi salti, nessuna azione).
    """
    reward = 0

    # Penalità se troppi salti in breve tempo (consumo fame inutile)
    if sum(jump_window) > 20: 
        reward -= 0.02
       
    # Penalità per inattività totale
    is_active = any(np.any(v == 1) if isinstance(v, np.ndarray) else v == 1 for v in action.values())
    if not is_active:
        reward -= 0.01

    return reward

def normalize(tensor):
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

# --- MAIN LOOP ---

def main(model, weights, env_name, n_episodes=3, max_steps=2000, show=True):
    env = aicrowd_gym.make(env_name)
    
    # Caricamento modello e pesi
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)
    print("Modello caricato con successo.")

    # Setup Grafico
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Reward Cumulativa in Tempo Reale")
    ax.set_xlabel("Passi")
    ax.set_ylabel("Reward")
    line, = ax.plot([], [], label="Reward Episodio")
    plt.legend()
    plt.grid(True)

    # Congelamento parametri tranne pi_head
    for param in agent.policy.parameters():
        param.requires_grad = False
    for param in agent.policy.pi_head.parameters():
        param.requires_grad = True

    # Setup Ottimizzatore
    optimizer = th.optim.RMSprop(
        filter(lambda p: p.requires_grad, agent.policy.parameters()), 
        lr=0.00001, alpha=0.99, eps=1e-8
    )
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    gamma = 0.99 

    cumulative_rewards = [] 
    
    # Creazione cartelle output
    base_dir = "./data/Stats/RLTranding_reward"
    os.makedirs(base_dir, exist_ok=True)

    # --- INIZIO EPISODI ---
    for episode in range(n_episodes):
        print(f"\n=== AVVIO EPISODIO {episode + 1}/{n_episodes} ===")
        obs = env.reset()
        
        # Inizializza il registro del "massimo posseduto"
        best_inventory = {key: 0 for key in MATERIAL_REWARDS.keys()}
        
        jump_window = deque(maxlen=20)
        cumulative_episode_reward = 0
        
        # Liste per grafici e statistiche
        episode_rewards = []
        steps_plot = []
        
        # Buffer per PPO
        batch_rewards = []
        batch_log_probs = []
        batch_advantages = []

        episode_dir = f"{base_dir}/episode{episode + 1}"
        os.makedirs(episode_dir, exist_ok=True)

        for step in range(max_steps):
            # 1. Scelta Azione
            action = agent.get_action(obs)
            action["ESC"] = 0 # Evita di aprire il menu di pausa
            
            # Esplorazione casuale ridotta
            if np.random.rand() < 0.05:
                action = env.action_space.sample()
                action["ESC"] = 0

            # 2. Esecuzione Step
            obs, _, done, _ = env.step(action)
            
            # 3. Aggiornamenti Variabili Stato
            jump_window.append(action.get("jump", 0))
            inventory = obs["inventory"]

            # 4. Calcolo Reward (Logica corretta)
            # Calcola reward materiali e aggiorna best_inventory se necessario
            material_reward = compute_reward_and_update_best(inventory, best_inventory)
            
            # Calcola reward azioni (movimento, salti)
            act_reward = action_based_reward(action, jump_window)
            
            reward = material_reward + act_reward
            cumulative_episode_reward += reward

            # 5. Controllo Obiettivo Raggiunto (Win Condition)
            if inventory.get("wooden_pickaxe", 0) > 0:
                print("\n\n!!! OBIETTIVO RAGGIUNTO: PICCONE DI LEGNO COSTRUITO !!!\n")
                reward += 500 # Bonus massiccio
                cumulative_episode_reward += 500
                done = True # Termina episodio
                
            # 6. Aggiornamento Grafico e Log
            steps_plot.append(step + 1)
            episode_rewards.append(cumulative_episode_reward)
            
            if step % 10 == 0: # Aggiorna grafico meno frequentemente per velocità
                line.set_xdata(steps_plot)
                line.set_ydata(episode_rewards)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
                plt.pause(0.001)
                print(f"\rStep: {step} | Reward Step: {reward:.4f} | Totale: {cumulative_episode_reward:.2f}", end="")

            # 7. Preparazione PPO
            # Recupera tensori necessari per il calcolo della loss
            agent_obs = agent._env_obs_to_agent(obs)
            da = agent.policy.get_output_for_observation(
                agent_obs,
                agent.policy.initial_state(1),
                th.tensor([False])
            )[0]
            ac = agent._env_action_to_agent(action, to_torch=True, check_if_null=False)
            log_prob = agent.policy.get_logprob_of_action(da, ac)
            
            # Calcolo Advantage
            advantage = reward + gamma * log_prob.detach().mean() - log_prob.mean()
            advantage = normalize(advantage)

            batch_rewards.append(reward)
            batch_log_probs.append(log_prob)
            batch_advantages.append(advantage)

            # 8. Aggiornamento PPO (Ogni 64 step o se done)
            if (step + 1) % 64 == 0 or done:
                batch_log_probs_t = th.stack(batch_log_probs)
                batch_advantages_t = th.stack(batch_advantages)

                # Calcolo Loss con Clipping
                r_t = th.exp(batch_log_probs_t - batch_log_probs_t.detach())
                epsilon = 0.2
                clipped_ratio = th.clamp(r_t, 1 - epsilon, 1 + epsilon)
                loss = -th.min(r_t * batch_advantages_t, clipped_ratio * batch_advantages_t).mean()

                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Reset batch
                batch_rewards = []
                batch_log_probs = []
                batch_advantages = []
                
                if step % 64 == 0:
                    print(f"\n[PPO UPDATE] Loss: {loss.item():.4f}")

            if show:
                env.render()
            
            if done:
                break

        # --- FINE EPISODIO ---
        print(f"\nEpisodio {episode + 1} terminato. Reward Totale: {cumulative_episode_reward}")
        cumulative_rewards.append(cumulative_episode_reward)

        # Salvataggio dati episodio
        df = pd.DataFrame({"Passo": steps_plot, "Reward Cumulativa": episode_rewards})
        df.to_excel(os.path.join(episode_dir, f"episode_{episode + 1}_rewards.xlsx"), index=False)

        plt.figure()
        plt.plot(steps_plot, episode_rewards, marker='o', label=f"Episode {episode + 1}")
        plt.title(f"Reward Ep {episode + 1}")
        plt.savefig(os.path.join(episode_dir, f"episode_{episode + 1}_graph.png"))
        plt.close()

    env.close()

    # --- SALVATAGGIO FINALE ---
    # Grafico complessivo
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_episodes + 1), cumulative_rewards, marker='o')
    plt.title("Reward Totale per Episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Reward Totale")
    plt.savefig(f"{base_dir}/cumulative_reward_trend_all.png")
    
    # Salvataggio pesi
    state_dict = agent.policy.state_dict()
    th.save(state_dict, "ppo_wooden_pickaxe.weights")
    print("Addestramento completato e pesi salvati.")

if __name__ == "__main__":
    parser = ArgumentParser("PPO MineRL - Wooden Pickaxe Training")

    parser.add_argument("--env", type=str, required=True, help="Nome dell'ambiente MineRL")
    parser.add_argument("--model", type=str, required=True, help="Percorso file .model")
    parser.add_argument("--weights", type=str, required=True, help="Percorso file .weights")
    parser.add_argument("--episodes", type=int, default=10, help="N. episodi")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max passi per episodio")
    parser.add_argument("--show", action="store_true", help="Render video")

    args = parser.parse_args()

    main(
        model=args.model,
        weights=args.weights,
        env_name=args.env, # Nota: ho rinominato l'argomento in main per chiarezza
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        show=args.show
    )