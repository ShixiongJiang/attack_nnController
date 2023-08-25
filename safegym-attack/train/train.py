import gymnasium
import safety_gymnasium
from gym.spaces import Discrete, Dict, Box

import numpy as np
import math
from copy import deepcopy


class SafetyPointGoal1(gymnasium.Env):
    def __init__(self, config=None):
        # super(SafetyPointGoal1, self).__init__()
        env_id = 'SafetyPointGoal1-v0'
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode='rgb_array')
        self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self.env.observation_space
        low = self.observation_space.low.astype('float32')
        high = self.observation_space.high.astype('float32')
        # self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype='float32')
        # print(self.action_space.low)
        self.action_space = gymnasium.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
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
        action[0] = action[0] / 20
        # print(action)
        obs, rew, done,truncated, info = self.env.step(action)
        goal_dist = 3 - 3 *max(obs[12:28])
        hazard_dist = 3 - 3 *max(obs[28:44])
        # print(hazard_dist)
        reward = self.radius * 2 - goal_dist
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
            final_reward = 20
            self.reset()
        if hazard_dist < 0.2:
            done = True
            final_reward = -2
            self.reset()
        if truncated:
            final_reward = -3
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


# env_id = 'SafetyPointGoal1-v0'
# safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode='human')
# gymnasium_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
#
# safety_gymnasium_env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(gymnasium_env)
# safe_env = safety_gymnasium.make(
#     'SafetyPointGoal1-v0')  # step returns (next_obervation, reward, cost, terminated, truncated, info)
# env = gymnasium.make('SafetyPointGoal1Gymnasium-v0',
#                      render_mode='human')  # step returns (next_obervation, reward, terminated, truncated, info)
from stable_baselines3 import A2C, SAC, PPO

env = SafetyPointGoal1()

model = PPO("MlpPolicy", env, verbose=0)
# model = PPO.load('SafetyPointGoal1-PPO-1.zip', env=env)
model.learn(total_timesteps=1000000)

# vec_env = model.get_env()
# obs = env.reset()
# env.render("human")
# print(vec_env.unwrapped.unwrapped.st)
model.save('SafetyPointGoal1-PPO-2.zip')
# for i in range(5000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)


