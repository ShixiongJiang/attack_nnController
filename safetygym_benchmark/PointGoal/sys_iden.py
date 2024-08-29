import random


import numpy as np
from safetygym_benchmark.PointGoal.pointgoal_env import PointGoal1
from safetygym_benchmark.PointGoal.attack_methods import get_op_action
from stable_baselines3 import PPO, SAC
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = PointGoal1()
obs, info = env.reset()
object = env.env.unwrapped._get_task()
model = PPO.load('./model/adv_PPO_PointGoal.zip')

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
action = [0, 0]

for i in range(100000):
    # Predict the action using the model (with some added noise for exploration)
    action, _state = model.predict(obs, deterministic=True)
    action = action + np.random.normal(size=2) * 0.2  # Add noise to the action
    # action = get_op_action(_state=obs, _action=[0.0, 0.0], previous_u = action, A=env.A, B=env.B)


    action = np.clip(action, -1, 1)
    # Take a step in the environment
    obs_next, reward, done, trun, info = env.step(action)

    obs_data.append(obs)
    obs_next_data.append(obs_next)
    u_data.append(action)
    sdot_data.append((obs_next - obs) / dt)
    su_data.append(np.concatenate([obs, action]))
    # print(np.concatenate([obs, action]).shape)
    # Update the current observation
    obs = obs_next

    # Reset the environment if done or truncated
    if done or trun:
        obs, info = env.reset()
        action = [0, 0]



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


A = coefficients[:, :44]  # First 44 columns correspond to the state
B = coefficients[:, 44:]  # Last 2 columns correspond to the action


intercept = model.intercept_






# mat = np.linalg.lstsq(su_data, sdot_data)[0].T
# A = mat[:, :44]
# B = mat[:, 44:]

# Save the estimated matrices A and B to a file
if not os.path.exists('./data'):
    os.mkdir('./data')
with open('./data/estimated_model_pointgoal.npz', 'wb') as f:
    # np.savez(f, A=A, B=B)
    np.savez(f, A=A, B=B)
print(A.shape)
print(B.shape)
print('Done')


