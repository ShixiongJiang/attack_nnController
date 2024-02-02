import random
import numpy as np
import pandas as pd

import gymnasium
import safety_gymnasium
from gym.spaces import Discrete, Dict, Box
import numpy as np
import math
from copy import deepcopy
from stable_baselines3 import A2C, PPO
from gym import spaces
def get_op_action(obs):
    obstacle = np.argmax(obs[28:44])

    turn = obs[4]
    velo = obs[3]
    if 4 < obstacle < 12:
        action0 = -1
    else:
        action0 = 1
    if 4 < obstacle < 12:
        if obstacle < 8:
            action1 = -1 * (8 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
        elif obstacle > 8:
            action1 = 1 * (obstacle - 8) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
        else:
            action1 = 0
    else:
        if obstacle <= 4:
            action1 = 1 * (obstacle + 1) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
        elif obstacle >= 13 and obstacle < 16:
            action1 = -1 * (15 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
        else:
            action1 = 0
    action = [action0, action1]
    return action
class SafetyPointGoal1(gymnasium.Env):
    def __init__(self, config=None):
        # super(SafetyPointGoal1, self).__init__()
        self.hazard_dist = None
        self.goal_dist = None
        env_id = 'SafetyPointGoalHazard0-v0'
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode='None')
        self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
        # This default action sapce is wrong
        self.action_space = self.env.action_space
        # print(type(self.action_space))
        self.action_space = gymnasium.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        # print(type(self.action_space))
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
        # action[0] = action[0] / 20
        obs, rew, done,truncated, info = self.env.step(action)
        goal_dist = 3 - 3 *max(obs[12:28])
        # hazard_dist = 3 - 3 *max(obs[28:44])
        self.goal_dist = goal_dist
        # self.hazard_dist = hazard_dist
        # print(hazard_dist)
        reward = self.radius - goal_dist
        # obs_reward = hazard_dist - self.radius
        self.reward_cache.append(reward)
        # self.avoid_reward_cache.append(obs_reward)
        if self.steps < 10:
            reach_reward = max(self.reward_cache)
        else:
            reach_reward = max(self.reward_cache[-10:])
        # if self.steps < 10:
        #     avoid_reward = min(self.avoid_reward_cache)
        # else:
        #     avoid_reward = min(self.avoid_reward_cache[-10:])
        # final_reward = min(reach_reward, avoid_reward)
        final_reward = reach_reward
        self.final_reward_cache.append(final_reward)
        if goal_dist < 0.4:
            done = True
            final_reward = 10
            self.reset()
        # if hazard_dist < 0.2:
        #     done = True
        #     self.reset()
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
# env.render("human")
i = 0
reach = 0
sdot_data = []
su_data = []
dt = 0.002
model = PPO.load('train/model/SafetyPointGoal1-PPO-7.zip', env=env)

for i in range(200000):

    # action = [(random.random() - 0.5) * 2,(random.random() - 0.5) * 2]
    action, _ = model.predict(obs)
    # action = get_op_action(obs)
    # print(action)
    obs_next, reward, done, trun, info = env.step(action)
    hazard_dist = 3 - 3 * max(obs_next[28:44])
    # if hazard_dist > 2.9:
    #     done = True
    if not done and not trun:
        sdot_data.append((obs_next - obs) / dt)
        su_data.append(np.concatenate([obs, action]))
        obs = obs_next
    else:
        obs, info = env.reset()


sdot_data = np.array(sdot_data)
su_data = np.array(su_data)
print(su_data.shape)
print(sdot_data.shape)
mat = np.linalg.lstsq(su_data, sdot_data)[0].T
A = mat[:, :44]
B = mat[:, 44:]
import os
if not os.path.exists('./data'):
    os.mkdir('./data')
f = open('./data/estimated_model_pointgoal.npz', 'wb')
np.savez(f, A=A, B=B)
print(B)
print('Done')


