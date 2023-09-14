import gymnasium
import safety_gymnasium
from gym.spaces import Discrete, Dict, Box

import numpy as np
import math
from copy import deepcopy
from stable_baselines3 import A2C, PPO
class AdvSafetyPointGoal1(gymnasium.Env):
    def __init__(self, config=None):
        # super(SafetyPointGoal1, self).__init__()
        env_id = 'SafetyPointGoal1-v0'
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode='human')
        self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self.env.observation_space
        low = self.observation_space.low.astype('float32')
        high = self.observation_space.high.astype('float32')
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype='float32')
        # print(self.observation_space)
        self.radius = 0.2
        self.reward_cache = []
        self.avoid_reward_cache = []
        self.final_reward_cache = []
        self.steps = 0


    # def _get_obs(self):
    #     return {"agent": self._agent_location, "target": self._target_location}
    #
    # def _get_info(self):
    #     return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset()

        return obs, info

    def step(self, action):
        self.steps += 1
        obs, rew, done,truncated, info = self.env.step(action)
        goal_dist = 3 - 3 *max(obs[12:28])
        hazard_dist = 3 - 3 *max(obs[28:44])

        # vice verse with original train
        obs_reward = self.radius - hazard_dist
        # self.reward_cache.append(reward)
        self.avoid_reward_cache.append(obs_reward)
        # if self.steps < 10:
        #     reach_reward = max(self.reward_cache)
        # else:
        #     reach_reward = max(self.reward_cache[-10:])
        if self.steps < 10:
            avoid_reward = min(self.avoid_reward_cache)
        else:
            avoid_reward = min(self.avoid_reward_cache[-10:])
        # final_reward = min(reach_reward, avoid_reward)
        final_reward = avoid_reward
        self.final_reward_cache.append(final_reward)
        if goal_dist < 0.4:
            done = True
            # final_reward = 20
            self.reset()
        if hazard_dist < 0.2:
            done = True
            final_reward = 5
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

env = AdvSafetyPointGoal1()

model = PPO.load('model/Adv_SafetyPointGoal1-PPO.zip', env=env)
# env = model.get_env()
obs, info = env.reset()
# env.render("human")
i = 0
reach = 0

for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    # action =
    # action = [0, 1]

    obs, reward, done, trun, info = env.step(action)
    # print(obs[0:12])
    if done:
        goal_dist = 3 - 3 * max(obs[12:28])
        print(goal_dist)
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
