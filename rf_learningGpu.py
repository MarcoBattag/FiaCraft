from argparse import ArgumentParser
import pickle
import os
import torch as th
import numpy as np
import sys
import aicrowd_gym
import minerl
from collections import deque
from openai_vpt.lib.tree_util import tree_map

# Setup percorsi
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

try:
    from openai_vpt.agent import MineRLAgent
except ImportError:
    print("\n[ERRORE] Libreria 'openai_vpt' non trovata.")
    sys.exit(1)

# --- CONTROLLO GPU ---
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"[*] Utilizzando il dispositivo: {device}")
if device.type == 'cuda':
    print(f"[*] GPU rilevata: {th.cuda.get_device_name(0)}")

MATERIAL_REWARDS = {
    "log": 2.0,            
    "planks": 4.0,         
    "stick": 6.0,
    "crafting_table": 20.0,
    "wooden_pickaxe": 500.0,
    "dirt": -0.2,
    "sand": -0.2,
    "gravel": -0.2 
}

def compute_reward_and_stop(inventory_obs, best_inventory, is_first_step=False):
    reward = 0.0
    task_done = False
    
    raw_data = inventory_obs[0] if isinstance(inventory_obs, list) else inventory_obs
    if raw_data is None: return 0.0, False

    if is_first_step or "_last_inv" not in best_inventory:
        best_inventory["_last_inv"] = {}

    for full_item_name, val in raw_data.items():
        quantity = int(np.max(val))
        category = next((c for c in MATERIAL_REWARDS if c in full_item_name), None)

        if not category: continue

        prev_q = best_inventory["_last_inv"].get(full_item_name, 0)

        if quantity > prev_q:
            diff = quantity - prev_q
            current_max = best_inventory.get(category, 0)
            
            if quantity > current_max:
                base_reward = MATERIAL_REWARDS[category]
                
                if category == "log":
                    if best_inventory.get("planks", 0) > 0: gain = float(diff * 0.01)
                    elif quantity > 4: gain = float(diff * (base_reward * 0.1))
                    else: gain = float(diff * base_reward)
                elif category == "planks":
                    if best_inventory.get("stick", 0) > 0: gain = float(diff * 0.1)
                    elif quantity > 8: gain = float(diff * (base_reward * 0.2))
                    else: gain = float(diff * base_reward)
                elif category == "stick":
                    if quantity > 4: gain = float(diff * (base_reward * 0.2))
                    else:
                        bonus = 10.0 if current_max == 0 else 0.0
                        gain = float(diff * base_reward + bonus)
                else:
                    gain = float(diff * base_reward)

                reward += gain
                best_inventory[category] = quantity
                print(f"\n[REWARD] {full_item_name}: {quantity} | +{gain:.2f}")
                
                if category == "wooden_pickaxe":
                    task_done = True

        elif quantity < prev_q:
            if category == "crafting_table":
                place_reward = 50.0
                reward += place_reward
                print(f"\n[EVENTO] Crafting Table PIAZZATA! +{place_reward}")

        best_inventory["_last_inv"][full_item_name] = quantity
                
    return reward, task_done

def normalize(tensor):
    if tensor.numel() <= 1: return tensor
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

def main(model_path, weights_path, env_name, n_episodes=25, max_steps=5000, show=True):
    # Creazione cartella checkpoint se manca
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    try:
        import envs 
        env = aicrowd_gym.make(env_name)
    except Exception as e:
        print(f"Errore caricamento ambiente {env_name}: {e}")
        sys.exit(1)

    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    
    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    
    # Caricamento pesi con gestione errore
    try:
        agent.load_weights(weights_path)
        print(f"[*] Pesi caricati correttamente da {weights_path}")
    except FileNotFoundError:
        print(f"[!] ATTENZIONE: File {weights_path} non trovato. Inizio con pesi base.")

    agent.policy.to(device)
    
    for param in agent.policy.parameters(): param.requires_grad = False
    for param in agent.policy.pi_head.parameters(): param.requires_grad = True

    optimizer = th.optim.RMSprop(filter(lambda p: p.requires_grad, agent.policy.parameters()), lr=0.0001)
    best_overall_score = -float('inf')

    for episode in range(n_episodes):
        obs = env.reset()
        agent.hidden_state = tree_map(lambda x: x.to(device) if x is not None else None, agent.policy.initial_state(1))
        
        best_inventory = {k: 0 for k in MATERIAL_REWARDS.keys()}
        cumulative_reward = 0.0
        batch_log_probs, batch_advantages = [], []

        print(f"\n--- Episodio {episode + 1} ---")

        for step in range(max_steps):
            action = agent.get_action(obs)
            if "ESC" not in action: action["ESC"] = 0
            
            next_obs, _, done, _ = env.step(action)
            
            inv_obs = next_obs.get("inventory", None)
            reward_step, task_completed = compute_reward_and_stop(inv_obs, best_inventory, (step==0))
            cumulative_reward += reward_step
            
            print(f"\rStep: {step:4d} | Rew: {reward_step:.2f} | Tot: {cumulative_reward:.2f}", end="", flush=True)

            # Training su GPU
            agent_obs = tree_map(lambda x: x.to(device) if isinstance(x, th.Tensor) else x, agent._env_obs_to_agent(obs))
            agent_action = tree_map(lambda x: x.to(device) if isinstance(x, th.Tensor) else x, agent._env_action_to_agent(action, to_torch=True))
            
            pi_logits, _, _ = agent.policy.get_output_for_observation(agent_obs, agent.hidden_state, th.tensor([False], device=device))
            log_prob = agent.policy.get_logprob_of_action(pi_logits, agent_action)
            
            batch_log_probs.append(log_prob)
            batch_advantages.append(th.tensor([reward_step], device=device, dtype=th.float32))

            if (step + 1) % 64 == 0 or done or task_completed:
                if len(batch_log_probs) > 0:
                    # Spostiamo tutto su GPU esplicitamente per il calcolo perdita
                    adv_t = normalize(th.stack(batch_advantages).to(device))
                    probs_t = th.stack(batch_log_probs).to(device)
                    
                    loss = -(probs_t * adv_t.detach()).mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    batch_log_probs, batch_advantages = [], []

            obs = next_obs
            if show: env.render()
            if done or task_completed: break

        # Salvataggio coerente col tuo comando
        if cumulative_reward > best_overall_score:
            best_overall_score = cumulative_reward
            save_path = "checkpoints/best_wooden_pickaxe.weights"
            th.save(agent.policy.state_dict(), save_path)
            print(f"\n[SALVATO] Nuovo record: {best_overall_score:.2f} -> {save_path}")

    env.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--env", type=str, default="FIA-WoodenPickaxe-v0")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.model, args.weights, args.env, args.episodes, args.max_steps, args.show)