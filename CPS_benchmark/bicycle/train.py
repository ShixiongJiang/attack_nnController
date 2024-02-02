# Training
from stable_baselines3 import SAC, PPO
import numpy as np
from CPS_benchmark.bicycle.bicycle_env import bicycleEnv, adv_bicycleEnv, laa_bicycleEnv

reached = []

env = bicycleEnv()
adv_env = adv_bicycleEnv()
laa_env = laa_bicycleEnv()

# print('Start training with SAC ...')
# learning_rate = 1e-3, n_steps = 1024, tune these
model = SAC("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=500000, progress_bar=False)
model.save('./model/SAC_bicycle.zip')

model = SAC("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=100000, progress_bar=False)
model.save('./model/surro_SAC_bicycle.zip')

model = PPO("MlpPolicy", adv_env, verbose=0)
model.learn(total_timesteps=100000, progress_bar=False)
model.save('./model/adv_PPO_bicycle.zip')

model = PPO("MlpPolicy", laa_env, verbose=0)
model.learn(total_timesteps=100000, progress_bar=False)
model.save('./model/laa_PPO_bicycle.zip')
