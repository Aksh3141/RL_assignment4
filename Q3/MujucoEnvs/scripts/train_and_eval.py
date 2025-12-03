import os
import sys
import json
import time
import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Fix Python path so config.py becomes importable
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)      # .../MujucoEnvs
sys.path.append(PROJECT_ROOT)

from config import configs


def make_gif(model, env_id, out_path, episodes=5):
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frame = env.render()
            frames.append(frame)

    env.close()
    imageio.mimsave(out_path, frames, fps=30)


def train_and_eval(env_name, algo_name):

    print(f"==============================")
    print(f"Training {algo_name} on {env_name}")
    print(f"==============================")

    # Load hyperparams from config
    hparams = configs[env_name][algo_name]["hyperparameters"]

    total_timesteps = hparams["total_timesteps"]
    run_name = hparams["Run_name"]

    # Directories
    log_dir = f"logs/{algo_name}/{env_name}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    results_txt = os.path.join(log_dir, "results.txt")

    # Store config used
    with open(os.path.join(log_dir, "config_used.json"), "w") as f:
        json.dump(hparams, f, indent=4)

    # Environment
    env = gym.make(env_name)

    # SB3 does NOT accept these keys in the model constructor
    NON_SB3_KEYS = [
        "total_timesteps",
        "algorithm",
        "log_interval",
        "eval_freq",
        "save_freq",
        "Run_name",
        "n_envs"
    ]

    filtered_hparams = {
        k: v for k, v in hparams.items() if k not in NON_SB3_KEYS
    }

    # Select algorithm
    ModelClass = PPO if algo_name == "PPO" else A2C

    # Create model
    model = ModelClass(
        "MlpPolicy",
        env,
        verbose=1,
        **filtered_hparams
    )

    # Training parameters
    save_freq = hparams["save_freq"]
    eval_freq = hparams["eval_freq"]

    # TRAIN LOOP
    for step in range(0, total_timesteps, eval_freq):

        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)

        # Save checkpoints
        if step % save_freq == 0:
            save_path = os.path.join(model_dir, f"model_{step}.zip")
            model.save(save_path)

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)

    # Evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Make GIF
    gif_path = os.path.join(log_dir, f"{algo_name}_{env_name}_eval.gif")
    make_gif(model, env_name, gif_path)

    # Save results
    with open(results_txt, "w") as f:
        f.write(f"Environment: {env_name}\n")
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Total timesteps: {total_timesteps}\n\n")

        f.write("===== EVALUATION =====\n")
        f.write(f"Mean Reward: {mean_reward}\n")
        f.write(f"Std Reward : {std_reward}\n\n")

        f.write("===== HYPERPARAMETERS USED =====\n")
        for k, v in hparams.items():
            f.write(f"{k}: {v}\n")

        f.write("\nModel saved at: " + final_model_path)
        f.write("\nGIF saved at  : " + gif_path)
        f.write("\n")

    print(f"Training + eval finished for {algo_name} on {env_name}")
    print(f"Results saved in: {log_dir}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    args = parser.parse_args()

    train_and_eval(args.env_name, args.algo)
