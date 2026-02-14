from argparse import ArgumentParser
import pickle
import time
import os
from torchvision import transforms
import gym
import minerl
import torch as th
import numpy as np

# Suppress FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from openai_vpt.agent import MineRLAgent
from data_loader import DataLoader
from openai_vpt.lib.tree_util import tree_map

# --- CONFIGURAZIONE ---
USING_FULL_DATASET = True
EPOCHS = 10 if USING_FULL_DATASET else 4
BATCH_SIZE = 16  
N_WORKERS = 16
DEVICE = "cuda"

REPORT_RATE = 10
SAVE_RATE = 50
LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.0
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0
MAX_BATCHES = 200000 if USING_FULL_DATASET else int(1e9)

# Augmentation (richiede C,H,W)
aug_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    ], p=0.5)
])

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def behavioural_cloning_train(data_dir, in_model, in_weights, out_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    env = gym.make("MineRLObtainDiamondShovel-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    original_agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    original_agent.load_weights(in_weights)
    env.close()

    policy = agent.policy
    original_policy = original_agent.policy

    for param in policy.parameters():
        param.requires_grad = False
    
    trainable_parameters = []
    for param in policy.pi_head.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    
    for param in policy.net.lastlayer.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)

    optimizer = th.optim.Adam(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_BATCHES)

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
        seq_len=32 
    )

    start_time = time.time()
    episode_hidden_states = {}
    
    acc_total_loss = 0
    acc_nll_loss = 0
    acc_kl_loss = 0
    best_loss_so_far = float('inf')
    
    print(f"--- TRAINING AVVIATO (FIXED) ---")
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | Workers: {N_WORKERS}")
    print("-" * 60)

    for batch_i, (batch_images, batch_actions, batch_episode_id, batch_first) in enumerate(data_loader):
        optimizer.zero_grad() 
        
        batch_total_loss = 0
        batch_nll = 0
        batch_kl = 0
        
        for image_seq, action_seq, episode_id, first_seq in zip(batch_images, batch_actions, batch_episode_id, batch_first):
            if image_seq is None: continue

            if episode_id not in episode_hidden_states:
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            seq_loss = 0
            seq_nll = 0
            seq_kl = 0
            seq_len = len(image_seq)
            valid_steps = 0 
            
            for t in range(seq_len):
                img_t = image_seq[t] 
                act_t = action_seq[t]
                is_first_t = first_seq[t]

                if is_first_t:
                    agent_state = policy.initial_state(1)

                agent_action = agent._env_action_to_agent(act_t, to_torch=True, check_if_null=True)
                if agent_action is None: continue
                
                valid_steps += 1

                # 1. Carichiamo su GPU e mettiamo in formato Canali-Prima (C, H, W) per l'Augmentation
                img_tensor = th.from_numpy(img_t).to(DEVICE).permute(2, 0, 1).float() / 255.0
                
                # 2. Augmentation
                img_tensor = aug_transform(img_tensor)
                
                # 3. FIX: Torniamo a Canali-Ultimi (H, W, C) perché l'Agente VPT si aspetta questo!
                img_tensor = img_tensor.permute(1, 2, 0)
                img_tensor = img_tensor * 255.0

                # 4. Creiamo il dizionario con la chiave corretta "img"
                agent_obs = {"img": img_tensor.unsqueeze(0)}
                
                first_tensor = th.from_numpy(np.array((is_first_t,))).to(DEVICE)

                pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                    agent_obs, agent_state, first_tensor
                )

                with th.no_grad():
                    original_pi_distribution, _, _ = original_policy.get_output_for_observation(
                        agent_obs, agent_state, first_tensor
                    )

                log_prob = policy.get_logprob_of_action(pi_distribution, agent_action)
                kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)
                
                current_nll = -log_prob
                step_loss = current_nll + KL_LOSS_WEIGHT * kl_div
                
                seq_loss += step_loss
                seq_nll += current_nll
                seq_kl += kl_div
                
                agent_state = new_agent_state

            if valid_steps > 0:
                final_loss = (seq_loss / valid_steps) / BATCH_SIZE
                final_loss.backward()
                
                batch_total_loss += final_loss.item()
                batch_nll += (seq_nll / valid_steps).item() / BATCH_SIZE
                batch_kl += (seq_kl / valid_steps).item() / BATCH_SIZE

            episode_hidden_states[episode_id] = tree_map(lambda x: x.detach(), agent_state)

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        acc_total_loss += batch_total_loss * BATCH_SIZE
        acc_nll_loss += batch_nll * BATCH_SIZE
        acc_kl_loss += batch_kl * BATCH_SIZE
        
        if batch_i % REPORT_RATE == 0 and batch_i > 0:
            time_since_start = time.time() - start_time
            avg_loss = acc_total_loss / REPORT_RATE
            avg_nll = acc_nll_loss / REPORT_RATE
            avg_kl = acc_kl_loss / REPORT_RATE
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"[Batch {batch_i}] Time: {time_since_start:.0f}s | "
                  f"Loss: {avg_loss:.4f} (NLL: {avg_nll:.4f} + KL: {avg_kl:.4f}) | "
                  f"LR: {current_lr:.6f}")
            
            acc_total_loss = 0
            acc_nll_loss = 0
            acc_kl_loss = 0

            if avg_loss < best_loss_so_far:
                best_loss_so_far = avg_loss
                best_path = out_weights.replace(".weights", "_best.weights")
                th.save(policy.state_dict(), best_path)

        if batch_i % SAVE_RATE == 0 and batch_i > 0:
            latest_path = out_weights.replace(".weights", "_latest.weights")
            th.save(policy.state_dict(), latest_path)

        if batch_i > MAX_BATCHES:
            break

    th.save(policy.state_dict(), out_weights)
    print("Training completato.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--in-model", required=True, type=str)
    parser.add_argument("--in-weights", required=True, type=str)
    parser.add_argument("--out-weights", required=True, type=str)

    args = parser.parse_args()
    behavioural_cloning_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)