import os
import random
import warnings
import pickle
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

import gymnasium as gym

warnings.filterwarnings("ignore", category=DeprecationWarning)
try:

    gym.logger.set_level(40) 
except Exception:
    pass

ENV_NAME = "InvertedPendulum-v4"
SEED = 42

SAVE_DIR = "saved_models"
TRAJ_DIR = "trajectories"
PLOTS_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


HIDDEN = 128
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
GAMMA = 0.99


TRAIN_TARGET_MIN = 400.0
TRAIN_TARGET_MAX = 500.0
MAX_TRAIN_EPISODES = 5000
TRAIN_BATCH = 5  


NUM_TRAJECTORIES = 500
MAX_EPISODE_LENGTH = 1000

SAMPLE_SIZES = [20,30,40,50,60,70,80,90,100]
REPS = 10

Trajectory = namedtuple("Trajectory", ["obs", "actions", "rewards", "log_probs", "dones"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        # learnable per-action log std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, obs_np):
       
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mean, std = self.forward(obs_t)
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(axis=-1)

        # DETACH before converting to numpy
        a_np = a.detach().cpu().numpy().flatten()
        logp_f = float(logp.detach().cpu().numpy())
        return a_np, logp_f

    def log_probs_batch(self, obs_t, act_t):
        mean, std = self.forward(obs_t)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(act_t).sum(axis=-1)
        return logp

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def compute_returns(rewards, gamma=GAMMA):
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns

def run_episode(env, policy, max_steps=MAX_EPISODE_LENGTH):
    obs, _ = env.reset()
    obs_list, action_list, reward_list, logp_list, done_list = [], [], [], [], []
    for _ in range(max_steps):
        action, logp = policy.sample_action(obs)
        obs_list.append(np.copy(obs))
        action_list.append(np.copy(action))
        logp_list.append(float(logp))  # already a float
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_list.append(float(reward))
        done_list.append(done)
        if done:
            break
    return Trajectory(obs=np.array(obs_list),
                      actions=np.array(action_list),
                      rewards=np.array(reward_list),
                      log_probs=np.array(logp_list),
                      dones=np.array(done_list))

# Training functions
def train_no_baseline(env_name, save_path, seed=SEED, batch_size=TRAIN_BATCH):
    seed_everything(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, act_dim).to(DEVICE)
    opt = optim.Adam(policy.parameters(), lr=ACTOR_LR)

    running_returns = deque(maxlen=100)
    ep_count = 0
    pbar = trange(MAX_TRAIN_EPISODES, desc="train_no_baseline")
    while ep_count < MAX_TRAIN_EPISODES:
        # collect batch_size episodes
        batch = []
        for _ in range(batch_size):
            tr = run_episode(env, policy)
            batch.append(tr)
            running_returns.append(float(tr.rewards.sum()))
            ep_count += 1

        total_loss = 0.0
        for tr in batch:
            obs_t = torch.as_tensor(tr.obs, dtype=torch.float32, device=DEVICE)
            act_t = torch.as_tensor(tr.actions, dtype=torch.float32, device=DEVICE)
            logp = policy.log_probs_batch(obs_t, act_t)  
            returns = compute_returns(tr.rewards)  # reward-to-go vector (T,)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
            total_loss = total_loss - (logp * returns_t).sum()
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if len(running_returns) == 100:
            avg = float(np.mean(running_returns))
            if TRAIN_TARGET_MIN <= avg <= TRAIN_TARGET_MAX:
                print(f"[no_baseline] reached avg return {avg:.2f} at episode {ep_count}")
                break
        pbar.update(0) 

    torch.save(policy.state_dict(), save_path)
    env.close()
    return save_path

def train_avg_baseline(env_name, save_path, seed=SEED+10, batch_size=TRAIN_BATCH):
    seed_everything(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, act_dim).to(DEVICE)
    opt = optim.Adam(policy.parameters(), lr=ACTOR_LR)

    running_returns = deque(maxlen=100)
    ep_count = 0
    while ep_count < MAX_TRAIN_EPISODES:
        batch = []
        for _ in range(batch_size):
            tr = run_episode(env, policy)
            batch.append(tr)
            running_returns.append(float(tr.rewards.sum()))
            ep_count += 1

        ep_returns = np.array([float(tr.rewards.sum()) for tr in batch], dtype=np.float32)
        b = float(ep_returns.mean())

        total_loss = 0.0
        for tr in batch:
            obs_t = torch.as_tensor(tr.obs, dtype=torch.float32, device=DEVICE)
            act_t = torch.as_tensor(tr.actions, dtype=torch.float32, device=DEVICE)
            logp = policy.log_probs_batch(obs_t, act_t)
            returns = compute_returns(tr.rewards)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
            advantages = returns_t - b
            total_loss = total_loss - (logp * advantages).sum()
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if len(running_returns) == 100:
            avg = float(np.mean(running_returns))
            if TRAIN_TARGET_MIN <= avg <= TRAIN_TARGET_MAX:
                print(f"[avg_baseline] reached avg return {avg:.2f} at episode {ep_count}")
                break

    torch.save(policy.state_dict(), save_path)
    env.close()
    return save_path

def train_reward_to_go_baseline(env_name, save_path, seed=SEED+20, batch_size=TRAIN_BATCH):
    seed_everything(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, act_dim).to(DEVICE)
    opt = optim.Adam(policy.parameters(), lr=ACTOR_LR)

    running_returns = deque(maxlen=100)
    ep_count = 0
    while ep_count < MAX_TRAIN_EPISODES:
        batch = []
        for _ in range(batch_size):
            tr = run_episode(env, policy)
            batch.append(tr)
            running_returns.append(float(tr.rewards.sum()))
            ep_count += 1

        # compute returns vectors for each episode
        returns_list = [compute_returns(tr.rewards) for tr in batch]
        max_len = max(len(r) for r in returns_list)
        b_t = np.zeros(max_len, dtype=np.float32)
        counts = np.zeros(max_len, dtype=np.int32)
        for r in returns_list:
            L = len(r)
            b_t[:L] += r
            counts[:L] += 1
        counts = np.maximum(counts, 1)
        b_t = b_t / counts  

        total_loss = 0.0
        for idx, tr in enumerate(batch):
            L = len(tr.rewards)
            obs_t = torch.as_tensor(tr.obs, dtype=torch.float32, device=DEVICE)
            act_t = torch.as_tensor(tr.actions, dtype=torch.float32, device=DEVICE)
            logp = policy.log_probs_batch(obs_t, act_t)
            returns_t = torch.as_tensor(returns_list[idx], dtype=torch.float32, device=DEVICE)
            baseline_vec = torch.as_tensor(b_t[:L], dtype=torch.float32, device=DEVICE)
            advantages = returns_t - baseline_vec
            total_loss = total_loss - (logp * advantages).sum()
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if len(running_returns) == 100:
            avg = float(np.mean(running_returns))
            if TRAIN_TARGET_MIN <= avg <= TRAIN_TARGET_MAX:
                print(f"[reward_to_go_baseline] reached avg return {avg:.2f} at episode {ep_count}")
                break

    torch.save(policy.state_dict(), save_path)
    env.close()
    return save_path

def train_value_baseline(env_name, policy_save_path, critic_save_path, seed=SEED+30, batch_size=TRAIN_BATCH):
    seed_everything(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, act_dim).to(DEVICE)
    critic = ValueNetwork(obs_dim).to(DEVICE)
    policy_opt = optim.Adam(policy.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    running_returns = deque(maxlen=100)
    ep_count = 0
    while ep_count < MAX_TRAIN_EPISODES:
        batch = []
        for _ in range(batch_size):
            tr = run_episode(env, policy)
            batch.append(tr)
            running_returns.append(float(tr.rewards.sum()))
            ep_count += 1

        total_policy_loss = 0.0
        total_critic_loss = 0.0
        for tr in batch:
            obs_t = torch.as_tensor(tr.obs, dtype=torch.float32, device=DEVICE)
            act_t = torch.as_tensor(tr.actions, dtype=torch.float32, device=DEVICE)
            returns = compute_returns(tr.rewards)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
            vvals = critic(obs_t)
            advantages = returns_t - vvals.detach()
            logp = policy.log_probs_batch(obs_t, act_t)
            total_policy_loss = total_policy_loss - (logp * advantages).sum()
            total_critic_loss = total_critic_loss + nn.MSELoss()(vvals, returns_t)

        policy_opt.zero_grad()
        total_policy_loss.backward()
        policy_opt.step()

        critic_opt.zero_grad()
        total_critic_loss.backward()
        critic_opt.step()

        if len(running_returns) == 100:
            avg = float(np.mean(running_returns))
            if TRAIN_TARGET_MIN <= avg <= TRAIN_TARGET_MAX:
                print(f"[value_baseline] reached avg return {avg:.2f} at episode {ep_count}")
                break

    torch.save(policy.state_dict(), policy_save_path)
    torch.save(critic.state_dict(), critic_save_path)
    env.close()
    return policy_save_path, critic_save_path

def collect_trajectories(env_name, policy_state_path, n_traj=NUM_TRAJECTORIES):
    seed_everything(SEED)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, act_dim).to(DEVICE)
    policy.load_state_dict(torch.load(policy_state_path, map_location=DEVICE))
    policy.eval()

    trajs = []
    for _ in tqdm(range(n_traj), desc=f"Collecting {os.path.basename(policy_state_path)}"):
        tr = run_episode(env, policy)
        trajs.append(tr)
    env.close()
    return trajs

# Gradient estimation utilities
def compute_policy_gradient_vector_from_trajs(policy, trajectories, baseline_type="no_baseline", critic=None):
    
    for p in policy.parameters():
        p.requires_grad = True
    policy.zero_grad()

    # compute baselines
    if baseline_type == "avg_baseline":
        ep_returns = np.array([float(tr.rewards.sum()) for tr in trajectories], dtype=np.float32)
        b_scalar = float(ep_returns.mean())
    else:
        b_scalar = None

    if baseline_type == "reward_to_go_baseline":
        returns_list = [compute_returns(tr.rewards) for tr in trajectories]
        max_len = max(len(r) for r in returns_list)
        b_t = np.zeros(max_len, dtype=np.float32)
        counts = np.zeros(max_len, dtype=np.int32)
        for r in returns_list:
            L = len(r)
            b_t[:L] += r
            counts[:L] += 1
        counts = np.maximum(counts, 1)
        b_t = b_t / counts
    else:
        b_t = None

    total_loss = 0.0
    for idx, tr in enumerate(trajectories):
        obs_t = torch.as_tensor(tr.obs, dtype=torch.float32, device=DEVICE)
        act_t = torch.as_tensor(tr.actions, dtype=torch.float32, device=DEVICE)
        logp = policy.log_probs_batch(obs_t, act_t)
        returns = compute_returns(tr.rewards)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

        if baseline_type == "no_baseline":
            advantages = returns_t
        elif baseline_type == "avg_baseline":
            advantages = returns_t - b_scalar
        elif baseline_type == "reward_to_go_baseline":
            L = len(returns)
            baseline_vec = torch.as_tensor(b_t[:L], dtype=torch.float32, device=DEVICE)
            advantages = returns_t - baseline_vec
        elif baseline_type == "value_baseline":
            if critic is None:
                raise ValueError("critic required for value_baseline")
            vvals = critic(obs_t).detach()
            advantages = returns_t - vvals
        else:
            raise ValueError("Unknown baseline_type")

        total_loss = total_loss - (logp * advantages).sum()

    total_loss.backward()

    grads = []
    for p in policy.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1))
    grad_vector = torch.cat(grads).detach().cpu().numpy()
    return grad_vector

def grad_l2_norm(grad_vector):
    return float(np.linalg.norm(grad_vector, ord=2))


# Main
def main():
    baseline_types = ["no_baseline", "avg_baseline", "reward_to_go_baseline", "value_baseline"]
    trained_paths = {}
    critic_paths = {}

    print("TRAINING POLICIES")
    trained_paths["no_baseline"] = train_no_baseline(ENV_NAME, os.path.join(SAVE_DIR, "no_baseline_policy.pt"))
    trained_paths["avg_baseline"] = train_avg_baseline(ENV_NAME, os.path.join(SAVE_DIR, "avg_baseline_policy.pt"))
    trained_paths["reward_to_go_baseline"] = train_reward_to_go_baseline(ENV_NAME, os.path.join(SAVE_DIR, "r2g_baseline_policy.pt"))
    policy_path, critic_path = train_value_baseline(ENV_NAME,
                                                   os.path.join(SAVE_DIR, "value_baseline_policy.pt"),
                                                   os.path.join(SAVE_DIR, "value_baseline_critic.pt"))
    trained_paths["value_baseline"] = policy_path
    critic_paths["value_baseline"] = critic_path

    print("\nCOLLECTING TRAJECTORIES")
    traj_files = {}
    for btype, path in trained_paths.items():
        trajs = collect_trajectories(ENV_NAME, path, n_traj=NUM_TRAJECTORIES)
        fn = os.path.join(TRAJ_DIR, f"{btype}_trajectories.pkl")
        with open(fn, "wb") as f:
            pickle.dump(trajs, f)
        traj_files[btype] = fn
        print(f"Saved {len(trajs)} trajectories for {btype} to {fn}")

    # load critic model for 'value_baseline' gradient computations
    critic_models = {}
    for b in baseline_types:
        if b == "value_baseline":
            env = gym.make(ENV_NAME)
            obs_dim = env.observation_space.shape[0]
            env.close()
            critic = ValueNetwork(obs_dim).to(DEVICE)
            critic.load_state_dict(torch.load(critic_paths["value_baseline"], map_location=DEVICE))
            critic.eval()
            critic_models[b] = critic
        else:
            critic_models[b] = None

    # load policy models for gradient computations
    policy_models = {}
    for b, path in trained_paths.items():
        env = gym.make(ENV_NAME)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        env.close()
        policy = GaussianPolicy(obs_dim, act_dim).to(DEVICE)
        policy.load_state_dict(torch.load(path, map_location=DEVICE))
        policy.eval()
        policy_models[b] = policy

    # load trajectories
    loaded_trajs = {}
    for b, fpath in traj_files.items():
        with open(fpath, "rb") as f:
            loaded_trajs[b] = pickle.load(f)

    # gradient estimation
    results = {b: {s: [] for s in SAMPLE_SIZES} for b in baseline_types}
    print("\nGRADIENT ESTIMATION")
    for b in baseline_types:
        print(f"Processing baseline: {b}")
        traj_pool = loaded_trajs[b]
        policy = policy_models[b]
        critic = critic_models[b]
        for s in SAMPLE_SIZES:
            norms = []
            for rep in range(REPS):
                sample = random.sample(traj_pool, s)
                grad_vec = compute_policy_gradient_vector_from_trajs(policy, sample, baseline_type=b, critic=critic)
                norms.append(grad_l2_norm(grad_vec))
            results[b][s] = norms
            print(f"  size {s}: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")

    # plotting
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    axes = axes.flatten()
    for idx, b in enumerate(baseline_types):
        ax = axes[idx]
        means = []
        stds = []
        for s in SAMPLE_SIZES:
            arr = np.array(results[b][s])
            means.append(arr.mean())
            stds.append(arr.std())
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(SAMPLE_SIZES, means, marker='o', label=f"{b} mean")
        ax.fill_between(SAMPLE_SIZES, means - stds, means + stds, alpha=0.3)
        ax.set_title(b)
        ax.set_xlabel("Sample size (trajectories)")
        ax.set_ylabel("Gradient L2 norm")
        ax.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "gradient_estimate_variance.png")
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}")

    # save numeric results
    with open(os.path.join(PLOTS_DIR, "gradient_estimate_results.pkl"), "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
