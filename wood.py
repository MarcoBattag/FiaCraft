from argparse import ArgumentParser
import numpy as np
import aicrowd_gym
import minerl

def main(env, n_episodes=3, max_steps=int(1e9), show=False):
    env = aicrowd_gym.make(env)

    for _ in range(n_episodes):
        obs = env.reset()
        for step in range(max_steps):
            # Azione hardcoded per spaccare legno
            action = env.action_space.noop()  # parte da noop
            action["forward"] = 1            # cammina avanti
            action["attack"] = 1             # colpisce il blocco
            action["jump"] = 0               # optional: 1 ogni tanto
            # piccolo movimento casuale della camera per colpire i blocchi diversi
            action["camera"] = np.array([0.0, np.random.uniform(-1, 1)])

            obs, _, done, _ = env.step(action)

            if show:
                env.render()  # render se richiesto

            if done:
                break

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run automatic Treechop on MineRL environment")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    args = parser.parse_args()

    main(args.env, show=args.show)
