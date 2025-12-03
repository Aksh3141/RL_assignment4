import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions
import torch.nn.functional as F

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
# from policies.experts import load_expert_policy
from functools import partial




class SBAgent(BaseAgent):
    def __init__(self, env_name, **hyperparameters):
        #implement your init function
        from stable_baselines3.common.env_util import make_vec_env
        #from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym
        
        self.hyperparameters=hyperparameters
        self.algorithm=self.hyperparameters['algorithm']
        self.env_name = env_name
        print(self.algorithm)
        
        self.env = make_vec_env(self.env_name) #initialize your environment. This variable will be used for evaluation. See train_sb.py
        Policy= "MlpPolicy"
        
        replay_buffer_class=None
        replay_buffer_kwargs=None
        if self.algorithm == 'PPO':
            from stable_baselines3 import PPO
            self.model = PPO(Policy, self.env,verbose=1, tensorboard_log="./data/")
        elif self.algorithm == 'A2C':
            from stable_baselines3 import A2C
            self.model = A2C(Policy, self.env,verbose=1, tensorboard_log="./data/")
        else:
            raise ValueError("Choose one of the following methods: 'A2C', 'PPO'")
        
    def learn(self):
        #implement your learn function. You should save the checkpoint periodically to <env_name>_sb3.zip
        from stable_baselines3.common.callbacks import CheckpointCallback,  ProgressBarCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
          save_freq=self.hyperparameters['save_freq'],
          save_path="./logs/",
          name_prefix=f"{self.algorithm}_{self.env_name}_{self.hyperparameters['Run_name']}",
          save_replay_buffer=True,
          save_vecnormalize=True,
        )

        eval_env= make_vec_env(self.env_name)
        eval_callback = EvalCallback(eval_env, best_model_save_path="../models/",
                             log_path="./data/", eval_freq=self.hyperparameters['eval_freq'],
                             deterministic=True, render=False)

        self.model.learn(total_timesteps=self.hyperparameters['total_timesteps'], log_interval=self.hyperparameters['log_interval'],
                         tb_log_name=f"{self.algorithm}_{self.env_name}_{self.hyperparameters['Run_name']}", progress_bar=True, 
                         callback= [eval_callback,checkpoint_callback])
        
    def load_checkpoint(self, checkpoint_path):
        #implement your load checkpoint function
        from stable_baselines3.common.vec_env import DummyVecEnv
        self.model = self.model.load(checkpoint_path)
    
    def get_action(self, observation):
        #implement your get action function
        action, _states = self.model.predict(observation)
        del _states
        return action, None