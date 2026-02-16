from argparse import ArgumentParser
import pickle
import os
import sys
import aicrowd_gym
import minerl
import torch as th

# --- AGGIUNTA: IMPORTA L'AMBIENTE PERSONALIZZATO ---
# Assumendo che il tuo ambiente sia definito in una cartella 'envs' nella root del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
try:
    import envs  # Questo file DEVE contenere gym.register(...) per FIA-WoodenPickaxe-v0
    print("[*] Ambienti personalizzati caricati con successo.")
except ImportError:
    print("[!] Attenzione: modulo 'envs' non trovato. Assicurati che sia nel PYTHONPATH.")

from openai_vpt.agent import MineRLAgent

def main(model, weights, env_name, n_episodes=3, max_steps=5000, show=False):
    # Controllo se l'ambiente esiste nel registro di gym
    print(f"[*] Inizializzazione ambiente: {env_name}")
    
    try:
        env = aicrowd_gym.make(env_name)
    except Exception as e:
        print(f"[ERRORE] Impossibile creare l'ambiente '{env_name}': {e}")
        print("[*] Suggerimento: Verifica di aver chiamato gym.register nel tuo file di definizione ambiente.")
        return

    # Caricamento parametri del modello
    if not os.path.exists(model):
        print(f"[ERRORE] File modello non trovato: {model}")
        return
        
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    
    # Inizializzazione Agente (spostato su GPU se disponibile)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"[*] Utilizzando dispositivo: {device}")
    
    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    for i in range(n_episodes):
        print(f"\n--- Inizio Episodio {i+1} ---")
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = agent.get_action(obs)
            
            # ESC non Ã¨ predetto dal modello, va forzato a 0
            action["ESC"] = 0
            
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if show:
                env.render()
                
            if steps % 100 == 0:
                print(f"Step: {steps}...", end="\r")

        print(f"\n--- Episodio {i+1} terminato dopo {steps} passi ---")
        
    env.close()

if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to .weights")
    parser.add_argument("--model", type=str, required=True, help="Path to .model")
    parser.add_argument("--env", type=str, required=True, help="Environment name (es. FIA-WoodenPickaxe-v0)")
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--episodes", type=int, default=3)

    args = parser.parse_args()

    main(args.model, args.weights, args.env, n_episodes=args.episodes, show=args.show)
