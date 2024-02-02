# Training
from stable_baselines3 import SAC, PPO
import numpy as np
from CPS_benchmark.DCmotor.DCmotor_env import DCEnv, adv_DCEnv, laa_DCEnv

reached = []

env = DCEnv()
adv_env = adv_DCEnv()
laa_env = laa_DCEnv()

# model = SAC("MlpPolicy", env, verbose=0)
# model.learn(total_timesteps=500000, progress_bar=False)
# model.save('./model/SAC_motor.zip')
#
# model = SAC("MlpPolicy", env, verbose=0)
# model.learn(total_timesteps=100000, progress_bar=False)
# # model.save('./model/SAC_bicycle.zip')
# model.save('./model/surro_SAC_motor.zip')

model = PPO("MlpPolicy", adv_env, verbose=0)
model.learn(total_timesteps=100000, progress_bar=False)
# model.save('./model/SAC_bicycle.zip')
model.save('./model/adv_PPO_motor.zip')

model = PPO("MlpPolicy", laa_env, verbose=0)
model.learn(total_timesteps=100000, progress_bar=False)
model.save('./model/laa_PPO_motor.zip')