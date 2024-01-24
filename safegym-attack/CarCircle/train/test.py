import math

from stable_baselines3 import A2C, SAC, PPO
import torch as th
from customerEnv import SafetyCarCircle
import numpy as np
env = SafetyCarCircle(render_mode='human')
# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=dict(pi=[128, 64, 64], vf=[64, 64]))

obs, info = env.reset()
model = PPO.load('model/SafetyCarCirclePPO-1.zip', env=env)
model = SAC.load('model/surro_SafetyCarCircleSAC-1.zip', env=env)


def get_op_action(obs, env):

    obstacle = np.argmax(obs[-16:])
    x = env.agent.pos[0]
    y = env.agent.pos[1]
    # print(x)
    vel = env.agent.vel
    u, v, _ = vel
    velo = math.sqrt(u**2 + v ** 2)
    action = [0, 0]
    if x >= 0 :
        if u >= 0:
            if v >= 0:
                action = [1 - abs(v) * 5, 1  ]
            else:
                action = [1,  1 - abs(v)* 5]
        else:
            if v >= 0:
                action = [-1, 1 ]
            else:
                action = [1, -1]
    else:
        if u <= 0:
            if v >= 0:
                action = [  1, 1 - abs(v) * 5  ]
            else:
                action = [ 1 - abs(v) * 5, 1]
        else:
            if v >= 0:
                action = [1, -1]

            else:
                action = [-1, 1]

    return action
for _ in range(1000):
    # action = env.action_space.sample()  # this is where you would insert your policy
    action, _state = model.predict(obs, deterministic=True)
    # action = get_op_action(obs)
    # action = [1, -1]
    obs, reward, terminated, truncated, info = env.step(action)
    # print(action)
    vel = env.agent.vel
    u, v, _ = vel
    # print(u, v)
    # print(reward)
    # observation, reward, terminated, truncated, info = env.step([0, 0])

    # print(reward)
    if terminated or truncated:
        obs, info = env.reset()
env.close()


