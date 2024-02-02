# Training
from stable_baselines3 import SAC, PPO
import numpy as np
from safetygym_benchmark.PointGoal.pointgoal_env import LaaPointGoal1, AdvPointGoal1, PointGoal1

reached = []

env = PointGoal1()
adv_env = AdvPointGoal1()
laa_env = LaaPointGoal1()

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000000, progress_bar=False)
model.save('./model/PPO_PointGoal.zip')

model = SAC("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1000000, progress_bar=False)
model.save('./model/surro_SAC_PointGoal.zip')

model = PPO("MlpPolicy", adv_env, verbose=0)
model.learn(total_timesteps=1000000, progress_bar=False)
model.save('./model/adv_PPO_PointGoal.zip')

model = PPO("MlpPolicy", laa_env, verbose=0)
model.learn(total_timesteps=1000000, progress_bar=False)
model.save('./model/laa_PPO_PointGoal.zip')