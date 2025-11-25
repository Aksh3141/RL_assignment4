import os
import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
import gymnasium as gym


def make_env(seed: int = 0, render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    env.reset(seed=seed)
    return env


def env_reset(env):
    obs, info = env.reset()
    return obs


def env_step(env, action):
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return next_obs, reward, done, info

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def dqn_update(q_net, target_net, optimizer, loss_fn, batch, gamma, device):
    """Standard DQN target: max over target network."""
    states, actions, rewards, next_states, dones = batch

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    # Q(s,a)
    q_values = q_net(states_t)
    q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # max_a' Q_target(s', a')
        next_q_target = target_net(next_states_t)
        max_next_q, _ = next_q_target.max(dim=1)
        target = rewards_t + gamma * (1.0 - dones_t) * max_next_q

    loss = loss_fn(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def double_dqn_update(q_net, target_net, optimizer, loss_fn, batch, gamma, device):
    """Double DQN target: argmax from online, value from target."""
    states, actions, rewards, next_states, dones = batch

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    # Q(s,a)
    q_values = q_net(states_t)
    q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Online network selects action
        next_q_online = q_net(next_states_t)
        next_actions = next_q_online.argmax(dim=1, keepdim=True)

        # Target network evaluates that action
        next_q_target = target_net(next_states_t)
        selected_next_q = next_q_target.gather(1, next_actions).squeeze(1)

        target = rewards_t + gamma * (1.0 - dones_t) * selected_next_q

    loss = loss_fn(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())



#  Agent wrapper
class Agent:
    def __init__(self,state_dim,action_dim,lr=1e-3,gamma=0.99,update_fn=None,target_update_freq=1000,device=None,):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.update_fn = update_fn
        self.target_update_freq = target_update_freq

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.train_steps = 0

    def act(self, state, epsilon: float = 0.0):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(q_values.argmax(dim=1).item())

    def q_values_numpy(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_network(state_t).cpu().numpy().squeeze(0)
        return q

    def update(self, replay_buffer: ReplayBuffer, batch_size: int):
        if len(replay_buffer) < batch_size:
            return None

        batch = replay_buffer.sample(batch_size)

        loss = self.update_fn(
            self.q_network,
            self.target_network,
            self.optimizer,
            self.loss_fn,
            batch,
            self.gamma,
            self.device,
        )

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss



#  Training & Evaluation
def train_agent(agent: Agent,env,replay_buffer: ReplayBuffer,num_episodes=600,max_steps_per_episode=1000,batch_size=64,min_replay_size=1000,
    epsilon_start=1.0,epsilon_end=0.05,epsilon_decay_episodes=400,agent_name="DQN",):

    rewards_per_episode = []

    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / max(epsilon_decay_episodes, 1)

    state = env_reset(env)

    # Fill initial replay buffer
    while len(replay_buffer) < min_replay_size:
        action = env.action_space.sample()
        next_state, reward, done, _ = env_step(env, action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env_reset(env)

    for ep in range(num_episodes):
        state = env_reset(env)
        ep_reward = 0.0

        for t in range(max_steps_per_episode):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env_step(env, action)

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            agent.update(replay_buffer, batch_size)

            if done:
                break

        rewards_per_episode.append(ep_reward)

        if epsilon > epsilon_end:
            epsilon = max(epsilon_end, epsilon - epsilon_decay)

        print(
            f"{agent_name} | Episode {ep+1}/{num_episodes} "
            f"Reward: {ep_reward:.2f} Epsilon: {epsilon:.3f}"
        )

    return rewards_per_episode


def evaluate_agent(agent: Agent, env, num_episodes=100, max_steps_per_episode=1000, agent_name="DQN"):
    returns = []
    q_values_per_action = {a: [] for a in range(env.action_space.n)}

    for ep in range(num_episodes):
        state = env_reset(env)
        ep_reward = 0.0

        for t in range(max_steps_per_episode):
            action = agent.act(state, epsilon=0.0)

            # record Q-values
            q_vals = agent.q_values_numpy(state)
            for a in range(env.action_space.n):
                q_values_per_action[a].append(q_vals[a])

            next_state, reward, done, _ = env_step(env, action)
            state = next_state
            ep_reward += reward
            if done:
                break

        returns.append(ep_reward)
        print(
            f"Eval {agent_name} | Episode {ep+1}/{num_episodes} Return: {ep_reward:.2f}"
        )

    returns = np.array(returns, dtype=np.float32)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    return mean_ret, std_ret, q_values_per_action

#  GIF Generation

def generate_gif(agent: Agent, filepath: str, max_steps=1000, seed: int = 0):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    env = make_env(seed=seed, render_mode="rgb_array")
    frames = []

    state = env_reset(env)
    for t in range(max_steps):
        frame = env.render()
        frames.append(frame)

        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, _ = env_step(env, action)
        state = next_state
        if done:
            break

    imageio.mimsave(filepath, frames, fps=30)
    env.close()

def main():
    # Folders
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("gif", exist_ok=True)

    # Reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_train_dqn = make_env(seed=seed)
    env_train_ddqn = make_env(seed=seed + 1)  
    env_eval = make_env(seed=seed + 2)

    state_dim = env_train_dqn.observation_space.shape[0]
    action_dim = env_train_dqn.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared hyperparameters
    lr = 1e-3
    gamma = 0.99
    replay_buffer_size = 100000
    batch_size = 64
    min_replay_size = 1000
    num_episodes = 800
    max_steps_per_episode = 1000
    target_update_freq = 1000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = 400

    # ----------------- DQN -----------------
    replay_buffer_dqn = ReplayBuffer(replay_buffer_size)
    dqn_agent = Agent(
        state_dim,
        action_dim,
        lr=lr,
        gamma=gamma,
        update_fn=dqn_update,
        target_update_freq=target_update_freq,
        device=device,
    )

    print("Training DQN...")
    dqn_rewards = train_agent(
        dqn_agent,
        env_train_dqn,
        replay_buffer_dqn,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        agent_name="DQN",
    )

    torch.save(dqn_agent.q_network.state_dict(), "models/dqn.pt")

    # ----------------- Double DQN -----------------
    replay_buffer_ddqn = ReplayBuffer(replay_buffer_size)
    ddqn_agent = Agent(
        state_dim,
        action_dim,
        lr=lr,
        gamma=gamma,
        update_fn=double_dqn_update,
        target_update_freq=target_update_freq,
        device=device,
    )

    print("Training Double DQN...")
    ddqn_rewards = train_agent(
        ddqn_agent,
        env_train_ddqn,
        replay_buffer_ddqn,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        agent_name="Double DQN",
    )

    torch.save(ddqn_agent.q_network.state_dict(), "models/ddqn.pt")

    # ----------------- Reward curves plot -----------------
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_rewards, label="DQN")
    plt.plot(ddqn_rewards, label="Double DQN")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("DQN vs Double DQN Training Rewards on LunarLander-v2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/reward curves.png")
    plt.close()

    # ----------------- Evaluation -----------------
    print("Evaluating DQN...")
    mean_dqn, std_dqn, qvals_dqn = evaluate_agent(
        dqn_agent, env_eval, num_episodes=100,
        max_steps_per_episode=max_steps_per_episode,
        agent_name="DQN",
    )

    print("Evaluating Double DQN...")
    mean_ddqn, std_ddqn, qvals_ddqn = evaluate_agent(
        ddqn_agent, env_eval, num_episodes=100,
        max_steps_per_episode=max_steps_per_episode,
        agent_name="Double DQN",
    )

    results = {
        "dqn": {
            "mean_reward": mean_dqn,
            "std_reward": std_dqn,
        },
        "double_dqn": {
            "mean_reward": mean_ddqn,
            "std_reward": std_ddqn,
        },
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation results:", results)

    # ----------------- Q-values per action plot (2x2) -----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    action_names = [
        "Action 0",
        "Action 1",
        "Action 2",
        "Action 3",
    ]  

    for a in range(action_dim):
        row = a // 2
        col = a % 2
        ax = axes[row, col]

        t_dqn = np.arange(len(qvals_dqn[a]))
        t_ddqn = np.arange(len(qvals_ddqn[a]))

        ax.plot(t_dqn, qvals_dqn[a], label="DQN")
        ax.plot(t_ddqn, qvals_ddqn[a], label="Double DQN", linestyle="--")

        ax.set_title(f"Q-values for {action_names[a]}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Q-value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/q values per action.png")
    plt.close()

    # ----------------- GIFs -----------------
    print("Generating GIF for DQN...")
    generate_gif(dqn_agent, "gif/dqn.gif", max_steps=1000, seed=seed + 3)

    print("Generating GIF for Double DQN...")
    generate_gif(ddqn_agent, "gif/ddqn.gif", max_steps=1000, seed=seed + 4)

    env_train_dqn.close()
    env_train_ddqn.close()
    env_eval.close()

if __name__ == "__main__":
    main()
