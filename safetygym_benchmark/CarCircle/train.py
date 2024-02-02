# Training
from stable_baselines3 import SAC, PPO
import numpy as np
from safetygym_benchmark.CarCircle.carcircle_env import CarCircleEnv, LaaCarCircleEnv, AdvCarCircleEnv

reached = []

env = CarCircleEnv()
adv_env = AdvCarCircleEnv()
laa_env = LaaCarCircleEnv()

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000000, progress_bar=False)
model.save('./model/PPO_carcircle.zip')

model = SAC("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1000000, progress_bar=False)
model.save('./model/surro_SAC_carcircle.zip')

model = PPO("MlpPolicy", adv_env, verbose=0)
model.learn(total_timesteps=1000000, progress_bar=False)
# model.save('./model/SAC_bicycle.zip')
model.save('./model/adv_PPO_carcircle.zip')

model = PPO("MlpPolicy", laa_env, verbose=0)
model.learn(total_timesteps=1000000, progress_bar=False)
model.save('./model/laa_PPO_carcircle.zip')