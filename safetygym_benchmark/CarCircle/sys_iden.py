import random


import numpy as np
from safetygym_benchmark.CarCircle.carcircle_env import CarCircleEnv
from stable_baselines3 import PPO, SAC
import os
from sklearn.linear_model import Ridge

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = CarCircleEnv()
obs, info = env.reset()
object = env.env.unwrapped._get_task()
model = PPO.load('./model/adv_PPO_carcircle.zip')

obs, info = env.reset()
# env.render("human")
i = 0
reach = 0
sdot_data = []
su_data = []
obs_data = []
obs_next_data = []
u_data = []
dt = 0.002

for i in range(100):

    # action = [(random.random() - 0.5) * 2,(random.random() - 0.5) * 2]
    action, _state = model.predict(obs, deterministic=True)
    action = action + np.random.normal(size=2) * 0.2
    pos = [env.agent.pos[0], env.agent.pos[1]]

    # print(f'accelaration: {obs[0:3]}, velo:{obs[3:6]}, magnetometer:{obs[9:12]}, action: {action}')
    obs_next, reward, done, trun, info = env.step(action)

    pos_next = [env.agent.pos[0], env.agent.pos[1]]

    obs_data.append(obs + pos)
    obs_next_data.append(obs_next + pos_next)
    u_data.append(action)
    sdot_data.append((obs_next - obs) / dt)
    su_data.append(np.concatenate([obs, action]))
    obs = obs_next
    if done or trun:
        obs, info = env.reset()
        # break

# Convert lists to numpy arrays
obs_data = np.array(obs_data)
obs_next_data = np.array(obs_next_data)
u_data = np.array(u_data)
sdot_data = np.array(sdot_data)
su_data = np.array(su_data)

XU_data = np.hstack((obs_data, u_data))


model = Ridge(alpha=1.0)


model.fit(XU_data, obs_next_data)


coefficients = model.coef_


A = coefficients[:, :42]  # First 44 columns correspond to the state
B = coefficients[:, 42:]  # Last 2 columns correspond to the action


intercept = model.intercept_






# mat = np.linalg.lstsq(su_data, sdot_data)[0].T
# A = mat[:, :44]
# B = mat[:, 44:]

# Save the estimated matrices A and B to a file
if not os.path.exists('./data'):
    os.mkdir('./data')
with open('./data/estimated_model_carcircle.npz', 'wb') as f:
    # np.savez(f, A=A, B=B)
    np.savez(f, A=A, B=B)
print(A.shape)
print(B.shape)
print('Done')

