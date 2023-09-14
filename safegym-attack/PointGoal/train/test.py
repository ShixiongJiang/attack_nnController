import random

import gymnasium
import safety_gymnasium
from gym.spaces import Discrete, Dict, Box

import numpy as np
import math
from copy import deepcopy
from stable_baselines3 import A2C, PPO
from gym import spaces
class SafetyPointGoal1(gymnasium.Env):
    def __init__(self, config=None):
        # super(SafetyPointGoal1, self).__init__()
        self.hazard_dist = None
        self.goal_dist = None
        env_id = 'SafetyPointGoalHazard1-v0'
        # env_id = 'SafetyPointGoal1-v0'
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode='human')
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
        action[0] = action[0] / 20
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
object = env.env.unwrapped._get_task()
# object_methods = [method_name for method_name in dir(object)
#                   if callable(getattr(object, method_name))]


# print(dir(object))
# config = env.env.unwrapped.config
# print(env.env.config)
# print(object._build_world_config(config))
# # print(dir(env.env))
# print(dir(env.env))
model = PPO.load('model/SafetyPointGoal1-PPO-5.zip', env=env)
# model = PPO.load('SafetyPointGoal1-PPO-1.zip', env=env)
# env = model.get_env()
obs, info = env.reset()
# env.render("human")
j = 0
reach = 0
y = []
u = []
for i in range(1000000):
    action, _state = model.predict(obs, deterministic=True)
    # action = [0.03, -1]
    # action =
    # action = [(random.random() - 0.5) * 2,(random.random() - 0.5) * 2]
    # print(f'accelaration: {obs[0:3]}, velo:{obs[3:6]}, magnetometer:{obs[9:12]}, action: {action}')
    obs, reward, done, trun, info = env.step(action)
#     if done:
#         goal_dist = 3 - 3 * max(obs[12:28])
#         # print(goal_dist)
#         if goal_dist <= 0.4:
#             reach +=1
#         j = j + 1
#     if j >= 100:
#         break
# print(reach)
    # if done or trun:
    #     break
    # y.append(obs[0:12])
    # u.append(action)
    # obstacle = np.argmax(obs[28:44])
    # # print(obstacle)
    # turn = obs[4]
    # velo = obs[3]
    # if obstacle > 4 and obstacle < 12:
    #     action0 = -1
    # else:
    #     action0 = 1
    # if obstacle > 4 and obstacle < 12:
    #     if obstacle < 8:
    #         action1 = -1 * (8 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
    #     elif obstacle > 8:
    #         action1 = 1 * (obstacle - 8) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
    #     else:
    #         action1 = 0
    # else:
    #     if obstacle <= 4 :
    #         action1 = 1 * (obstacle + 1) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
    #     elif obstacle >= 13 and obstacle < 16:
    #         action1 = -1 * (15 - obstacle ) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
    #     else:
    #         action1 = 0
    # # action = [action0, action1]
    # # print(max(obs[28:44]))
    #
    #
    #
    # # print(turn)
    # angular = obs[8]
    # velo =
    # print('+++++++++++++++++++++++++++++++++++++++++++')


    # print(env.env.metadata)
    # print(obs[6:9])
    # if done:
    #     goal_dist = 3 - 3 * max(obs[12:28])
    #     print(goal_dist)
# while i < 1000:
#
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#
#     if done:
#         i += 1
#
#
#         goal_dist = 3 - 3 * max(obs[0][12:28])
#         print(goal_dist)
#         hazard_dist = 3 - 3 * max(obs[0][28:44])
#         if goal_dist < 0.4:
#             reach += 1
#
# print(reach)

# import sysid
# import pylab as pl
# ss1 = sysid.StateSpaceDiscreteLinear(
#     A=np.matrix([[0,0.1,0.2],[0.2,0.3,0.4],[0.4,0.3,0.2]]),
#     B=np.matrix([[1,0],[0,1],[0,-1]]),
#     C=np.matrix([[1,0,0],[0,1,0]]), D=np.matrix([[0,0],[0,0]]),
#     Q=pl.diag([0.1,0.1,0.1]), R=pl.diag([0.04,0.04]), dt=0.1)
#
# tf =
#
# sysid.subspace_det_algo1(y=y, u=u, )