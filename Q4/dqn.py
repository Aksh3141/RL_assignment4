import torch
import torch.nn as nn
import torch.optim as optim
import random, numpy as np
import gymnasium as gym
from collections import deque
import imageio, os, json
import matplotlib.pyplot as plt
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENV_NAME = "LunarLander-v3"
GAMMA = 0.99
LR = 1e-3
BATCH = 64
BUFFER_SIZE = 100000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 50000
TARGET_UPDATE = 2000
MAX_EPISODES = 2000

SAVE_PATH = "Q4/checkpoints/dqn_lunar.pt"
GIF_PATH = "Q4/gifs/dqn_best.gif"
EVAL_JSON = "Q4/stats/dqn_eval.json"
TRAIN_STATS = "Q4/stats/dqn_train_stats.txt"
CONVERGENCE_TXT = "Q4/stats/dqn_convergence_info.txt"

os.makedirs("Q4/checkpoints", exist_ok=True)
os.makedirs("Q4/gifs", exist_ok=True)
os.makedirs("Q4/stats", exist_ok=True)
os.makedirs("Q4/plots", exist_ok=True)


# ---------------------------------
# Q-Network
# ---------------------------------
class QNet(nn.Module):
    def __init__(self, obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------
# Replay Buffer
# ---------------------------------
class Replay:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, *t):
        self.buf.append(t)

    def sample(self, B):
        batch = random.sample(self.buf, B)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(ns), np.array(d))

    def __len__(self):
        return len(self.buf)


# ---------------------------------
# Evaluation (GIF + JSON metrics)
# ---------------------------------
def evaluate_best_gif(q):
    print("\nEvaluating DQN and saving BEST GIF...")

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    rewards_list, best_reward, best_frames = [], -1e9, None

    q.eval()
    with torch.no_grad():
        for ep in range(5):
            frames = []
            s, _ = env.reset()
            done = False
            total_r = 0

            while not done:
                frames.append(env.render())
                a = q(torch.tensor(s, dtype=torch.float32, device=DEVICE)).argmax().item()
                s, r, term, trunc, _ = env.step(a)
                done = term or trunc
                total_r += r

            rewards_list.append(float(total_r))
            print(f"Eval Episode {ep+1}: reward={total_r:.1f}")

            if total_r > best_reward:
                best_reward = total_r
                best_frames = frames

    env.close()
    q.train()

    # Save GIF
    if best_frames is not None:
        imageio.mimsave(GIF_PATH, best_frames, fps=30)
        print(f"Saved BEST GIF → {GIF_PATH}")

    # Save JSON
    json.dump({
        "episode_rewards": rewards_list,
        "mean_reward": float(np.mean(rewards_list)),
        "std_reward": float(np.std(rewards_list)),
        "best_reward": float(best_reward)
    }, open(EVAL_JSON, "w"), indent=4)

    print(f"Metrics saved → {EVAL_JSON}")


# ---------------------------------
# Training
# ---------------------------------
def train_dqn():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q = QNet(obs_dim, n_actions).to(DEVICE)
    q_target = QNet(obs_dim, n_actions).to(DEVICE)
    q_target.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=LR)
    replay = Replay(BUFFER_SIZE)

    steps, eps = 0, EPS_START
    rewards = []
    losses = []
    epsilons = []

    train_stats = open(TRAIN_STATS, "w")

    start_time = time.time()

    for ep in range(1, MAX_EPISODES + 1):
        s, _ = env.reset()
        ep_r = 0

        while True:
            steps += 1

            # epsilon-greedy
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    a = q(torch.tensor(s, dtype=torch.float32, device=DEVICE)).argmax().item()

            ns, r, term, trunc, _ = env.step(a)
            d = term or trunc

            replay.push(s, a, r, ns, d)
            s = ns
            ep_r += r

            # epsilon decay
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-steps / EPS_DECAY)

            # training step
            if len(replay) >= BATCH:
                ss, aa, rr, nss, dd = replay.sample(BATCH)

                ss = torch.tensor(ss, dtype=torch.float32, device=DEVICE)
                aa = torch.tensor(aa, dtype=torch.int64, device=DEVICE)
                rr = torch.tensor(rr, dtype=torch.float32, device=DEVICE)
                nss = torch.tensor(nss, dtype=torch.float32, device=DEVICE)
                dd = torch.tensor(dd, dtype=torch.float32, device=DEVICE)

                q_val = q(ss).gather(1, aa.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    max_target = q_target(nss).max(1)[0]
                    tgt = rr + GAMMA * max_target * (1 - dd)

                loss = nn.MSELoss()(q_val, tgt)

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())

            if steps % TARGET_UPDATE == 0:
                q_target.load_state_dict(q.state_dict())

            if d:
                break

        rewards.append(ep_r)
        epsilons.append(eps)
        train_stats.write(f"{ep},{ep_r}\n")
        train_stats.flush()

        print(f"EP {ep}: Reward={ep_r:.1f}  Eps={eps:.3f}")

        # SOLVED condition
        if len(rewards) >= 100 and np.mean(rewards[-100:]) >= 200:
            print("\nSolved! Saving DQN model...")
            torch.save(q.state_dict(), SAVE_PATH)
            break

    train_stats.close()
    env.close()

    # -------------------------
    # Save convergence text summary
    # -------------------------
    total_time = time.time() - start_time
    with open(CONVERGENCE_TXT, "w") as f:
        f.write(f"DQN for {ENV_NAME}\n")
        f.write(f"Solved at episode: {ep}\n")
        f.write(f"Mean last 100 reward: {np.mean(rewards[-100:]):.2f}\n")
        f.write(f"Total steps: {steps}\n")
        f.write(f"Training time (sec): {total_time:.2f}\n")
        f.write(f"Training time (min): {total_time/60:.2f}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"Batch size: {BATCH}\n")
        f.write(f"Buffer size: {BUFFER_SIZE}\n")
        f.write(f"Target update: {TARGET_UPDATE}\n")
        f.write(f"Epsilon final: {eps:.4f}\n")

    print(f"\nConvergence info saved → {CONVERGENCE_TXT}")

    # -------------------------
    # Save plots
    # -------------------------
    if rewards:
        plt.figure()
        plt.plot(rewards)
        plt.title("Reward Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig("Q4/plots/dqn_rewards.png")

    if losses:
        plt.figure()
        plt.plot(losses)
        plt.title("Loss Curve")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.savefig("Q4/plots/dqn_loss.png")

    if epsilons:
        plt.figure()
        plt.plot(epsilons)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.savefig("Q4/plots/dqn_epsilon.png")

    # evaluate final policy
    evaluate_best_gif(q)


if __name__ == "__main__":
    train_dqn()
