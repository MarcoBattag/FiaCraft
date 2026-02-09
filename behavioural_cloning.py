from argparse import ArgumentParser
import pickle
import time
import os
from torchvision import transforms
import gym
import minerl
import torch as th
import numpy as np

# Rimosso import amp per evitare conflitti con VPT
from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from openai_vpt.lib.tree_util import tree_map

USING_FULL_DATASET = False

EPOCHS = 10 if USING_FULL_DATASET else 4
# Se hai 8GB VRAM, Batch size 4 in FP32 dovrebbe starci. Se va in OOM, scendi a 2.
BATCH_SIZE = 16 if USING_FULL_DATASET else 4 
N_WORKERS = 4 
DEVICE = "cuda"

# Augmentation: applicata probabilisticamente
aug_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    ], p=0.5)
])

LOSS_REPORT_RATE = 10
LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.0
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0
MAX_BATCHES = 2000 if USING_FULL_DATASET else int(1e9)

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

    # Create a copy which will have the original parameters
    original_agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    original_agent.load_weights(in_weights)
    env.close()

    policy = agent.policy
    original_policy = original_agent.policy

    # 1. Freeze di base
    for param in policy.parameters():
        param.requires_grad = False
    
    # 2. Unfreeze intelligente
    trainable_parameters = []
    
    # Sblocca Pi Head (decisioni finali)
    for param in policy.pi_head.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    
    # Sblocca ultimo layer visuale/processamento
    for param in policy.net.lastlayer.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)

    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

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
    
    loss_sum = 0
    
    print(f"Inizio training su dispositivo: {DEVICE} (FP32 Mode)")

    for batch_i, (batch_images, batch_actions, batch_episode_id, batch_first) in enumerate(data_loader):
        optimizer.zero_grad() 
        batch_loss = 0
        
        for image_seq, action_seq, episode_id, first_seq in zip(batch_images, batch_actions, batch_episode_id, batch_first):
            
            if image_seq is None:
                continue

            if episode_id not in episode_hidden_states:
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            seq_loss = 0
            seq_len = len(image_seq)
            valid_steps = 0 
            
            # --- LOOP TEMPORALE ---
            for t in range(seq_len):
                img_t = image_seq[t] 
                act_t = action_seq[t]
                is_first_t = first_seq[t]

                if is_first_t:
                    agent_state = policy.initial_state(1)

                agent_action = agent._env_action_to_agent(act_t, to_torch=True, check_if_null=True)
                
                if agent_action is None:
                    continue
                
                valid_steps += 1

                # Data Augmentation (Manual processing)
                if isinstance(img_t, np.ndarray):
                    # Conversione HWC -> CHW e normalizzazione
                    img_tensor = th.from_numpy(img_t).permute(2, 0, 1).float() / 255.0
                else:
                    img_tensor = img_t
                
                # Applica augmentation
                img_tensor = aug_transform(img_tensor)
                
                # Riconversione per VPT (vuole dizionario "pov")
                # Nota: VPT internamente gestisce la conversione, ma qui gli passiamo
                # un numpy array uint8 per compatibilità massima con _env_obs_to_agent
                img_aug_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                agent_obs = agent._env_obs_to_agent({"pov": img_aug_np})
                first_tensor = th.from_numpy(np.array((is_first_t,))).to(DEVICE)

                # Forward pass Standard (FP32) - Niente autocast qui
                pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    first_tensor
                )

                with th.no_grad():
                    original_pi_distribution, _, _ = original_policy.get_output_for_observation(
                        agent_obs,
                        agent_state,
                        first_tensor
                    )

                log_prob = policy.get_logprob_of_action(pi_distribution, agent_action)
                kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)
                
                step_loss = -log_prob + KL_LOSS_WEIGHT * kl_div
                seq_loss += step_loss
                
                agent_state = new_agent_state
            
            # --- FINE LOOP TEMPORALE ---

            if valid_steps > 0:
                final_loss = (seq_loss / valid_steps) / BATCH_SIZE
                final_loss.backward()
                batch_loss += final_loss.item()

            episode_hidden_states[episode_id] = tree_map(lambda x: x.detach(), agent_state)

        # Optimization step standard
        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        loss_sum += batch_loss
        
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Loss: {loss_sum / LOSS_REPORT_RATE:.4f}, LR: {current_lr:.6f}")
            loss_sum = 0
            
            ckpt_path = out_weights.replace(".weights", f"_latest.weights")
            th.save(policy.state_dict(), ckpt_path)

        if batch_i > MAX_BATCHES:
            break

    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)
    print("Training completato e modello salvato.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data")
    parser.add_argument("--in-model", required=True, type=str, help="Path to .model")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to .weights")
    parser.add_argument("--out-weights", required=True, type=str, help="Path to save weights")

    args = parser.parse_args()
    behavioural_cloning_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)