import gymnasium
import safety_gymnasium

import numpy as np
import math
from copy import deepcopy

from stable_baselines3 import A2C, SAC, PPO
import torch as th
from customerEnv import SafetyCarCircle
env = SafetyCarCircle()
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128, 128], vf=[64, 64]))
# model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
model = PPO.load('SafetyCarCirclePPO-1.zip', env=env)
# print(model.policy.net_arch)
model.set_env(env)
model.learn(total_timesteps=10000000)

# vec_env = model.get_env()
# obs = env.reset()
# env.render("human")
# print(vec_env.unwrapped.unwrapped.st)
model.save('SafetyCarCirclePPO-1.zip')