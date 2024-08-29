
from stable_baselines3 import SAC, PPO
import numpy as np
from safetygym_benchmark.PointGoal.pointgoal_env import LaaPointGoal1, AdvPointGoal1, PointGoal1
from attack_methods import get_op_action
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

reached = []
model = PPO.load('./model/PPO_PointGoal.zip')

env = PointGoal1()
epoch = 0
reach = 0
violate = 0
rho_list = []
perturb_rho_list = []
dis_list = []
total_epoch = 1
print(env.A.shape)
print(env.B.shape)

while True:
    # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
    obs, info = env.reset()
    _state = obs
    rho = []
    perturb_rho = []
    dis = []
    action = [0, 0]
    count = 0
    while True:
        goal_dist = 3 - 3 * max(obs[12:28])
        obs_dist = 3 - 3 * max(obs[28:44])
        dis.append(obs_dist)
        rho.append(-goal_dist)
        pertub_goal_dist = 3 - 3 * max(obs[12:28])
        perturb_rho.append(-pertub_goal_dist)

        # action, _state = model.predict(obs, deterministic=True)

        # action = get_op_action(_state=obs, _action=[0.0, 0.0], previous_u = action, A=env.A, B=env.B)

        action = get_op_action(_state=obs, _action=[0.0, 0.0], previous_u = action, A=env.A, B=env.B)
        print(action)



        obs, reward, done, trun, info = env.step(action)
        # print(obs[0:12])
        if done or trun:
            rho_list.append(rho)
            perturb_rho_list.append(perturb_rho)
            dis_list.append(dis)
            epoch +=1
            goal_dist = 3 - 3 * max(obs[12:28])
            obs_dist = 3 - 3 * max(obs[28:44])
            if goal_dist < 0.4:
                reach +=1
            elif obs_dist < 0.2:
                violate += 1
            break
    if epoch >= total_epoch:
        # print(rho)
        # print(perturb_rho)
        break
