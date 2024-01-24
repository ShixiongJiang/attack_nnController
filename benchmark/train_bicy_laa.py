from stable_baselines3 import SAC
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
import sys
import pandas as pd
import matplotlib.pyplot as plt
from numpy import cos, sin
from gym import spaces
from gym.error import DependencyNotInstalled
from typing import Optional
from control.matlab import ss, lsim, linspace, c2d
from functools import partial
# from state_estimation import Estimator
import math
import gym
from stable_baselines3 import PPO, SAC, TD3, DDPG, DQN, A2C
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# define CSTR model
def bicycle(x,t,u, params={}):
    lr = 1.105
    lf = 1.738
    psi = x[2]
    v = x[3]
    alpha = u[0]
    sigma = u[1]
    xdot =np.zeros(4)
    beta = math.atan((lr/(lr+lf)*math.tan(sigma)))
    xdot[0] = v*math.cos(psi+beta)
    xdot[1] = v*math.sin(psi+beta)
    xdot[2] = v/lr*math.sin(beta)
    xdot[3] = alpha
    return xdot
class baseline_bicycleEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        super(baseline_bicycleEnv).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        ob_high = np.ones(4) * 20
        action_high = np.ones(2) * 7
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-ob_high, high=ob_high, dtype=np.float32)
        self.step_const = 50
        self.freq = 0.1
        self.steps = 0
        self.center = [2, 2, 0, 2 * math.sqrt(2)]
        # self.obstacle =  np.array(([0.5,0.5,0,math.sqrt(2)/2], [1,1,0,math.sqrt(2)/2]))
        self.obstacle = np.array(
            ([-0.88615284, -1.00078591, -1.5150387, 2.41190424], [-1.06931684, 0.66430412, -2.53652435, 4.46120764]))
        self.reward_cache = []  # cache distances to target norm ball center
        self.final_reward_cache = []  # cache final reward
        self.state = np.random.rand(4) * 2 - 2.2
        self.horizon = np.arange(0, self.step_const + 2, 1) * 0.1
        self.target_norm_radius = 0.6
        self.safe_norm_radius = 0.4
        self.max_reward_list = []
        self.avoid_reward_cache = []
        self.quality_list = []
        self.total_steps = 0
        self.step_history = []
        self.k = 30

    def step(self, action):
        terminated = False
        ts = [self.horizon[self.steps], self.horizon[self.steps + 1]]
        self.state = odeint(bicycle, self.state, ts, args=(action,))[-1]
        dist = np.linalg.norm(self.state - self.center)
        obs_dist = min(np.linalg.norm(self.state - self.obstacle[0]), np.linalg.norm(self.state - self.obstacle[1]))
        reward = self.target_norm_radius - dist
        obs_reward = obs_dist - self.safe_norm_radius
        # reward = self.target_norm_radius - dist + obs_dist - 0.3
        self.reward_cache.append(reward)
        self.avoid_reward_cache.append(obs_reward)
        if self.steps < 10:
            reach_reward = max(self.reward_cache)
        else:
            reach_reward = max(self.reward_cache[-10:])
        if self.steps < 10:
            avoid_reward = min(self.avoid_reward_cache)
        else:
            avoid_reward = min(self.avoid_reward_cache[-10:])
        final_reward = - reward
        self.reward_cache.append(reward)
        self.steps += 1
        self.total_steps += 1

        # if dist <= self.target_norm_radius:
        #     final_reward = 100
        self.final_reward_cache.append(final_reward)
        if self.steps == self.step_const or obs_dist <= self.safe_norm_radius:
            # final_reward = -10
            self.max_reward_list.append(max(self.final_reward_cache))  # use max final reward to measure episodes
            self.step_history.append(self.total_steps)
            self.quality_list.append(sum(self.final_reward_cache))
            terminated = True
            # self.reset()

        # Return next state, reward, done, info
        return self.state, final_reward, terminated, {}

    def reset(self):
        self.steps = 0
        # self.state = np.random.rand(4)*2-1
        self.state = np.random.rand(4) * 2 - 2.2
        self.reward_cache = []
        self.step_const = self.k
        self.horizon = np.arange(0, self.step_const + 2, 1) * 0.1
        self.final_reward_cache = []
        return self.state  # reward, done, info can't be included

    def render(self, mode="human"):
        return

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# Training
reached = []
for k in [20]:
    env = baseline_bicycleEnv()
    env.k = k
    print('Start training with PPO ...')
    baseline_model = SAC("MlpPolicy", env, verbose=0)

    baseline_model.learn(total_timesteps=500000, progress_bar=False)
    # vec_env = baseline_model.get_env()

baseline_model.save('baseline_bicycle_model.zip')
print('done')