import pygame
from argparse import ArgumentParser
import pickle
import pandas as pd
import os
import aicrowd_gym
import minerl
import torch as th
import torch.nn.functional as F
import numpy as np
import cv2  # Importa OpenCV
from collections import deque

from openai_vpt.agent import MineRLAgent

# Nome del file Excel
EXCEL_FILE = "reward_human_feedback_log.xlsx"

# Configurazione per monitorare i salti
JUMP_WINDOW = 20  # Numero massimo di passi da considerare

def log_to_excel(step, cumulative_reward):
    # Controlla se il file esiste
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
    else:
        # Crea un nuovo DataFrame se il file non esiste
        df = pd.DataFrame(columns=["Step", "Cumulative Reward"])

    # Crea un DataFrame per i nuovi dati
    new_data = pd.DataFrame({"Step": [step], "Cumulative Reward": [cumulative_reward]})
    
    # Usa pd.concat per aggiungere i nuovi dati
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Scrivi i dati nel file Excel
    df.to_excel(EXCEL_FILE, index=False)

def calculate_advantage(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values)]
    return advantages, returns

def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=True):
    pygame.init()
    
    screen = pygame.display.set_mode((1280, 720))  # Finestra per catturare input
    pygame.display.set_caption("MineRL Human Feedback Viewer")
    
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)
    
    optimizer = th.optim.Adam(agent.policy.parameters(), lr=1e-4)
    epsilon = 0.2  # Fattore di clipping
    gamma = 0.99  # Discount factor
    lam = 0.95  # GAE lambda

    cumulative_reward = 0
    manual_reward = 0
    stop_training = False

    for _ in range(n_episodes):
        if stop_training:
            break

        obs = env.reset()
        done = False
        step = 0

        # Batch data
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_log_probs = []
        batch_values = []

        while not done and step < max_steps:
            # Input manuale per reward
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        manual_reward += 0.3
                        print(f"Manual reward increased: {manual_reward}")
                    elif event.key == pygame.K_MINUS:
                        manual_reward -= 0.3
                        print(f"Manual reward decreased: {manual_reward}")
                    elif event.key == pygame.K_q:
                        print("Training interrotto. Salvataggio del modello...")
                        stop_training = True
                        break

            if stop_training:
                break

            # Ottieni output dal modello
            output = agent.policy.get_output_for_observation(
                agent._env_obs_to_agent(obs),
                agent.policy.initial_state(1),
                th.tensor([False])
            )

            pd = output[0]  # Distribuzione delle probabilità
            value = output[1]  # Valore dello stato corrente

            # Campiona un'azione
            action = agent.get_action(obs)

            # Converte l'azione
            ac = agent._env_action_to_agent(action, to_torch=True, check_if_null=False)

            # Calcola log-probabilità
            log_prob = agent.policy.get_logprob_of_action(pd, ac)

            # Esegui l'azione nell'ambiente
            obs, reward, done, _ = env.step(action)

            # Accumula i dati
            batch_obs.append(obs)
            batch_actions.append(action)
            batch_rewards.append(reward + manual_reward)
            batch_log_probs.append(log_prob)
            batch_values.append(value)
            cumulative_reward += reward
            manual_reward = 0  # Resetta la reward manuale

            # Aggiorna ogni 32 passi
            # Modifica nella logica di ottimizzazione
            if (step + 1) % 32 == 0 or done:
                agent_input = agent._env_obs_to_agent(obs)
                output = agent.policy.get_output_for_observation(
                    agent_input,
                    agent.policy.initial_state(1),
                    th.tensor([False])
                )
                next_value = output[1]
                
                batch_values.append(next_value)

                # Calcolo vantaggi e returns
                advantages, _ = calculate_advantage(batch_rewards, batch_values)

                # Normalizzazione dei vantaggi
                batch_advantages = th.tensor(advantages)
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # Converti log-probabilità in tensori
                batch_log_probs_tensor = th.stack(batch_log_probs)

                # Calcolo del rapporto r_t e applicazione del clipping
                r_t = th.exp(batch_log_probs_tensor - batch_log_probs_tensor.detach())
                clipped_ratio = th.clamp(r_t, 1 - epsilon, 1 + epsilon)
                loss = -th.min(r_t * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Backpropagation e aggiornamento
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Resetta i batch
                batch_obs = []
                batch_actions = []
                batch_rewards = []
                batch_log_probs = []
                batch_values = []


            if show:
                image = obs["pov"]
                image = np.flip(image, axis=1)
                image = np.rot90(image)
                image = cv2.resize(image, (720, 1280))
                image = image.astype(np.uint8)

                surface = pygame.surfarray.make_surface(image)
                screen.blit(surface, (0, 0))
                pygame.display.update()

            step += 1

        print(f"Reward cumulativa per l'episodio: {cumulative_reward}")

    env.close()
    th.save(agent.policy.state_dict(), "ppo_human_feedback_weights.weights")
    print("Modello salvato.")
    pygame.quit()


if __name__ == "__main__":
    parser = ArgumentParser("Manual Reward Adjustment")

    parser.add_argument("--env", type=str, required=True, help="Nome dell'ambiente MineRL (es. MineRLObtainDiamondShovel-v0)")
    parser.add_argument("--model", type=str, required=True, help="Percorso al file '.model' pre-addestrato.")
    parser.add_argument("--weights", type=str, required=True, help="Percorso al file '.weights' pre-addestrato.")
    parser.add_argument("--episodes", type=int, default=1, help="Numero di episodi da eseguire.")
    parser.add_argument("--max-steps", type=int, default=1e9, help="Numero massimo di passi per episodio.")
    parser.add_argument("--show", action="store_true", help="Visualizza il rendering dell'ambiente.")

    args = parser.parse_args()

    main(
        model=args.model,
        weights=args.weights,
        env=args.env,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        show=args.show
    )
