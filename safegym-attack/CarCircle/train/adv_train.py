import gymnasium
import safety_gymnasium

import numpy as np
import math
from copy import deepcopy
from adv_env import adv_SafetyCarCircle
from stable_baselines3 import A2C, SAC, PPO
import torch as th
from customerEnv import SafetyCarCircle
env = adv_SafetyCarCircle()
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128, 128], qf=[64, 64]))
# model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
model = PPO.load('model/adv_SafetyCarCircle-PPO-1.zip', env=env)
# model = SAC("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)# print(model.policy.net_arch)
model.set_env(env)
model.learn(total_timesteps=400000)

# vec_env = model.get_env()
# obs = env.reset()
# env.render("human")
# print(vec_env.unwrapped.unwrapped.st)
model.save('model/adv_SafetyCarCircle-PPO-1.zip')