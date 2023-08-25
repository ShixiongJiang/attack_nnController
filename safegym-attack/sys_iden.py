import random

import numpy as np
from scipy.linalg import svd
from scipy.signal import lti, cont2discrete, dlti
import gymnasium
import safety_gymnasium
from gym.spaces import Discrete, Dict, Box

import numpy as np
import math
from copy import deepcopy
from stable_baselines3 import A2C, PPO
class SafetyPointGoal1(gymnasium.Env):
    def __init__(self, config=None):
        # super(SafetyPointGoal1, self).__init__()
        self.hazard_dist = None
        self.goal_dist = None
        env_id = 'SafetyPointGoal1-v0'
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode='rgb_array')
        self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.radius = 0.2
        self.reward_cache = []
        self.avoid_reward_cache = []
        self.final_reward_cache = []
        self.steps = 0
        self.done = False


    # def _get_obs(self):
    #     return {"agent": self._agent_location, "target": self._target_location}
    #
    # def _get_info(self):
    #     return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset()
        self.steps = 0
        self.done = True
        return obs, info

    def step(self, action):
        self.steps += 1
        obs, rew, done,truncated, info = self.env.step(action)
        goal_dist = 3 - 3 *max(obs[12:28])
        hazard_dist = 3 - 3 *max(obs[28:44])
        self.goal_dist = goal_dist
        self.hazard_dist = hazard_dist
        # print(hazard_dist)
        reward = self.radius - goal_dist
        obs_reward = hazard_dist - self.radius
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
        final_reward = min(reach_reward, avoid_reward)
        self.final_reward_cache.append(final_reward)
        if goal_dist < 0.4:
            done = True
            final_reward = 10
            self.reset()
        if hazard_dist < 0.2:
            done = True
            self.reset()
        if truncated:
            done = True
            self.reset()
        return obs, final_reward, done,truncated, info

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

env = SafetyPointGoal1()
obs, info = env.reset()
u_list = []
obs_list = []
u_lists = []
obs_lists = []

for i in range(10000):

    action = [random.Random(), random.Random()]

    obs, reward, done, trun, info = env.step(action)
    u_list.append(action)
    obs_list.append(obs[0:12])
    if done:
        u_lists.append(u_list)
        obs_lists.append(obs_list)
        u_list = []
        obs_list = []

# # Assuming you have collected data u and x
# u = np.array(...)  # Input data matrix (size: input_dimension x time_steps)
# x = np.array(...)  # Output data matrix (size: output_dimension x time_steps)
# # Construct Hankel matrices
def construct_hankel(data, n_lags):
    num_rows = data.shape[0]
    num_cols = data.shape[1] - n_lags + 1
    hankel_matrix = np.zeros((num_rows * n_lags, num_cols))
    for i in range(num_cols):
        hankel_matrix[:, i] = data[:, i:i + n_lags].flatten()
    return hankel_matrix

A_estimated = None
B_estimated = None
for i in range(len(u_lists)):
    u = u_lists[i]
    x = obs_lists[i]
    n_lags = 2  # Number of lags in the Hankel matrix
    H = construct_hankel(np.vstack((u, x)), n_lags)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = svd(H)

    # Determine system order (order of A matrix)
    sys_order = ...  # You need to determine this based on the singular values

    # Extract the relevant matrices from the SVD
    U_r = U[:, :sys_order]
    S_r = np.diag(S[:sys_order] ** 0.5)
    Vt_r = Vt[:sys_order, :]

    # Estimate the system matrices A and B
    A_estimated = U_r[sys_order:, :].dot(np.linalg.pinv(U_r[:-sys_order, :])).dot(Vt_r).dot(np.linalg.pinv(S_r))
    B_estimated = U_r[sys_order:, :].dot(np.linalg.pinv(S_r))

print("Estimated A:", A_estimated)
print("Estimated B:", B_estimated)
