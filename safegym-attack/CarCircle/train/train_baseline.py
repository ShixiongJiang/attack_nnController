import gymnasium
import safety_gymnasium

import numpy as np
import math
from copy import deepcopy


class baseline_Adv_SafetyCarCircle(gymnasium.Env):
    def __init__(self, render_mode='None'):
        # super(SafetyPointGoal1, self).__init__()
        env_id = 'SafetyCarCircle1-v0'
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode=render_mode)
        self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self.env.observation_space
        low = self.observation_space.low.astype('float32')
        high = self.observation_space.high.astype('float32')

        self.action_space = gymnasium.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.radius = 0.2
        self.reward_cache = []
        self.avoid_reward_cache = []

        self.final_reward_cache = []
        self.steps = 0

        self.agent = self.env.unwrapped._agent()

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
        action[1] = action[1] / 20
        # print(action)
        obs, rew, done,truncated, info = self.env.step(action)
        final_reward = 0
        x = self.agent.pos[0]
        y = self.agent.pos[1]

        radius = np.sqrt(x ** 2 + y ** 2)
        vel = self.agent.vel
        u, v, _ = vel
        # velocity = np.sqrt(u ** 2 + v ** 2)
        # print(velocity)
        velocity = (-u * y + v * x)
        # print(velocity)
        goal_reward = velocity - 4
        wall_reward = (1.12 - x) * 5
        circle_reward = min(1.49 - radius, radius - 1.2) * 5
        self.reward_cache.append(goal_reward)
        self.avoid_reward_cache.append(min(wall_reward, circle_reward))

        if self.steps < 10:
            reach_reward = max(self.reward_cache)
        else:
            reach_reward = max(self.reward_cache[-10:])
        if self.steps < 10:
            avoid_reward = min(self.avoid_reward_cache)
        else:
            avoid_reward = min(self.avoid_reward_cache[-10:])

        final_reward = -rew
        # if velocity > 4:
        #     final_reward = 5
        # if info['cost'] != 0:
        #     done = True
        #     # self.reset()
        #     final_reward = -10
        #
        # if 1.50 - radius < 0:
        #     final_reward = -10
            # self.reset()
        # self.final_reward_cache.append(final_reward)
        # if goal_dist < 0.4:
        #     done = True
        #     final_reward = 10
        #     self.reset()
        # if hazard_dist < 0.2:
        #     done = True
        #     final_reward = -1000
        #     self.reset()
        # if truncated:
        #     final_reward = 5
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


from stable_baselines3 import A2C, SAC, PPO

env = baseline_Adv_SafetyCarCircle()

model = PPO("MlpPolicy", env, verbose=0)
# model = PPO.load('Adv_SafetyPointGoal1-PPO.zip', env=env)
model.learn(total_timesteps=1000000)

model.save('model/baseline_SafetyCarCircle1-PPO.zip')
