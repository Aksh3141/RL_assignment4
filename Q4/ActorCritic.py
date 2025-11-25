"""
ActorCritic for LunarLander-v3 (TD(0) updates, per-step)
"""

import os
import time
import json
from collections import deque, namedtuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import imageio

# -------------------------
# Hyperparameters 
# -------------------------
ENV_NAME = "LunarLander-v3"
SEED = 1
HIDDEN_SIZE = 256
LR_ACTOR = 1e-4
LR_CRITIC = 5e-4
GAMMA = 0.99
ENTROPY_BETA = 0.001
MAX_EPISODES = 500000
UPDATE_EVERY_N_STEPS = 4096
TARGET_RUNNING_AVG = 260.0
LOG_EVERY_EPISODES = 10

SAVE_PATH = "Q4/checkpoints/a2c_lunar_lander.pt"
EVAL_JSON = "Q4/stats/a2c_eval.json"
GIF_PATH = "Q4/gifs/a2c_best.gif"

Transition = namedtuple("Transition", ("state", "action", "reward", "done", "placeholder"))

# -------------------------
# Utilities
# -------------------------
def set_seed(env, seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except TypeError:
        pass

# -------------------------
# Actor Network
# -------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Critic Network
# -------------------------
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# -------------------------
# Compute Returns (not used for TD updates, kept for compatibility)
# -------------------------
def compute_returns(rewards, dones, last_value, gamma=GAMMA):
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    R = float(last_value)
    for i in reversed(range(T)):
        R = rewards[i] + gamma * R * (1.0 - float(dones[i]))
        returns[i] = R
    return returns

# -------------------------
# Evaluation + GIF
# -------------------------
def evaluate_best_gif(actor, device):
    print("\nEvaluating and saving BEST GIF...")

    os.makedirs("Q4/gifs", exist_ok=True)
    os.makedirs("Q4/stats", exist_ok=True)

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    rewards_list = []
    best_reward = -1e9
    best_frames = None

    actor.eval()
    with torch.no_grad():
        for ep in range(5):
            frames = []
            s, _ = env.reset()
            done = False
            total_r = 0.0

            while not done:
                frames.append(env.render())

                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                logits = actor(s_t)
                # deterministic greedy action for eval GIF
                a = int(logits.argmax(dim=-1).item())

                s, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                total_r += float(r)

            rewards_list.append(total_r)
            print(f"Eval Episode {ep+1}: {total_r:.2f}")

            if total_r > best_reward:
                best_reward = total_r
                best_frames = frames

    if best_frames is not None and len(best_frames) > 0:
        imageio.mimsave(GIF_PATH, best_frames, fps=30)
        print(f"Saved GIF: {GIF_PATH}")

    env.close()
    actor.train()

    json.dump({
        "episode_rewards": rewards_list,
        "mean_reward": float(np.mean(rewards_list)) if rewards_list else 0.0,
        "std_reward": float(np.std(rewards_list)) if rewards_list else 0.0,
        "best_reward": float(best_reward) if best_reward > -1e9 else 0.0
    }, open(EVAL_JSON, "w"), indent=4)

# -------------------------
# Training Loop (TD updates per step)
# -------------------------
def train():
    env = gym.make(ENV_NAME)
    set_seed(env, SEED)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor(obs_dim, n_actions).to(device)
    critic = Critic(obs_dim).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    
    buffer = []
    episode_rewards = []
    running_avg_window = deque(maxlen=100)
    actor_losses, critic_losses = [], []

    total_steps = 0
    start_time = time.time()

    os.makedirs("Q4/checkpoints", exist_ok=True)
    os.makedirs("Q4/plots", exist_ok=True)
    os.makedirs("Q4/stats", exist_ok=True)

    for ep in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # Prepare tensors for current state
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # policy
            logits = actor(s_t)
            dist = Categorical(logits=logits)
            action_tensor = dist.sample()                  # tensor on device
            action = int(action_tensor.item())

            # value for current state
            v_s = critic(s_t)  # shape [1]

            # step env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            total_steps += 1

            # compute TD target: if terminal -> target = r, else r + gamma * V(next)
            if done:
                td_target = torch.tensor([reward], dtype=torch.float32, device=device)
            else:
                next_s_t = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                v_next = critic(next_s_t).detach()  # detach to avoid backprop through next state
                td_target = torch.tensor([reward], dtype=torch.float32, device=device) + GAMMA * v_next

            # advantage = TD_target - V(s)
            advantage = (td_target - v_s.detach())

            # Actor update (policy gradient with TD advantage)
            log_prob = dist.log_prob(action_tensor)  # shape [1]
        
            # negative sign because we minimize
            actor_loss = -(log_prob * advantage).mean() 

            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

            # Critic update (MSE to TD target)
            critic_loss = nn.functional.mse_loss(v_s, td_target)

            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

            try:
                actor_losses.append(float(actor_loss.item()))
            except Exception:
                actor_losses.append(0.0)
            try:
                critic_losses.append(float(critic_loss.item()))
            except Exception:
                critic_losses.append(0.0)

            buffer.append(Transition(state.copy(), int(action), float(reward), bool(done), None))
            if len(buffer) > UPDATE_EVERY_N_STEPS:
                buffer = buffer[-UPDATE_EVERY_N_STEPS:]

            state = next_state

        # book-keeping after episode ends
        episode_rewards.append(ep_reward)
        running_avg_window.append(ep_reward)
        running_avg = float(np.mean(running_avg_window))

        with open("Q4/stats/a2c_train_stats.txt", "a") as f:
            f.write(f"{ep},{ep_reward},{running_avg}\n")

        if ep % LOG_EVERY_EPISODES == 0:
            elapsed = time.time() - start_time
            print(f"Ep {ep}  Reward={ep_reward:.2f}  Avg100={running_avg:.2f}  Steps={total_steps}  Time={elapsed:.1f}s")

        # -------------------------
        # Solve Condition
        # -------------------------
        if running_avg >= TARGET_RUNNING_AVG and len(running_avg_window) >= 100:
            print(f"\nSolved at episode {ep}!")

            # Save convergence info
            total_time = time.time() - start_time
            info_path = "Q4/stats/a2c_convergence_info.txt"

            with open(info_path, "w") as f:
                f.write(f"Environment: {ENV_NAME}\n")
                f.write(f"Solved at episode: {ep}\n")
                f.write(f"Total steps: {total_steps}\n")
                f.write(f"Running average reward: {running_avg:.2f}\n")
                f.write(f"Total time (sec): {total_time:.2f}\n")
                f.write(f"Total time (min): {total_time/60:.2f}\n")
                f.write(f"Actor LR: {LR_ACTOR}\n")
                f.write(f"Critic LR: {LR_CRITIC}\n")
                f.write(f"Hidden size: {HIDDEN_SIZE}\n")
                f.write(f"Batch size (steps): {UPDATE_EVERY_N_STEPS}\n")

            print(f"Saved convergence info → {info_path}")
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, SAVE_PATH)
            break

    env.close()

    # plots (same style as original)
    if actor_losses:
        plt.figure()
        plt.plot(actor_losses)
        plt.title("Actor Loss")
        plt.savefig("Q4/plots/actor_loss.png")
        plt.close()

    if critic_losses:
        plt.figure()
        plt.plot(critic_losses)
        plt.title("Critic Loss")
        plt.savefig("Q4/plots/critic_loss.png")
        plt.close()

    if episode_rewards:
        plt.figure()
        ma = np.convolve(episode_rewards, np.ones(10)/10, mode="valid")
        plt.plot(episode_rewards, alpha=0.5)
        plt.plot(np.arange(9, 9+len(ma)), ma, label="MA10")
        plt.legend()
        plt.title("Reward Curve")
        plt.savefig("Q4/plots/reward_curve.png")
        plt.close()

    evaluate_best_gif(actor, device)

if __name__ == "__main__":
    train()
