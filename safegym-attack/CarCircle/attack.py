import math

import numpy as np

import torch
import torch.nn as nn
from train.customerEnv import SafetyCarCircle
from stable_baselines3 import PPO, SAC
from baseline import Gradient_attack, MAD, laa_attack


def black_attack(env, state, model, surro_model, adv_model, epsilon):
    action = surro_model.predict(state)[0]
    #     print(action)
    effect = None
    attack = None
    _action = action
    _state = state
    state_range = np.array([epsilon])

    op_action = adv_model.predict(state)[0]
    #     op_action = (result.x)
    state = torch.from_numpy(state)

    state = state.detach().unsqueeze(0).requires_grad_(True)
    effect = 1000
    for i in range(20):

        action = surro_model.policy._predict(state)[0].requires_grad_(True)

        action = action.double()

        # compute the distance
        loss = (torch.tensor([op_action]) - action).pow(2).sum().sqrt()
        # loss = mse(torch.tensor([op_action]).requires_grad_(True), action)
        surro_model.policy.zero_grad()
        loss = loss.double()
        # print(action.dtype)
        loss.backward()
        # print(loss)
        state_grad = state.grad.detach().squeeze()
        # print(state_grad[torch.argmax(state_grad[12:28])])
        # for i in range(12, 28):
        #     if state_grad[i] > 0:
        #         state_grad[i] = 0
        perturbed_state = state - state_grad.sign() * epsilon * 0.2
        l = torch.from_numpy(_state) - torch.from_numpy(state_range)
        u = torch.from_numpy(_state) + torch.from_numpy(state_range)
        perturbed_state = torch.max(torch.min(perturbed_state, u), l)
        state = perturbed_state.detach().requires_grad_(True)

        dist = 3 - max(_state[12:28]) * 3
        pertub_dist = 3 - max(state[0][12:28]) * 3

        if pertub_dist <= dist:
            #
            if loss <= effect:
                effect = loss
                # print(effect)
                attack = (state.detach().numpy() - _state)[0]
    # if pertub_obs_dist - 0.2 > 0:
    #
    #     if next_obs_dist < effect:
    #         effect = next_obs_dist
    # print(effect)
    # print('+++++++++++++++++++')
    if attack is None:
        return np.zeros_like(_state)
    else:
        # print('find solution')
        return attack

def get_op_action(env):

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


def white_attack(env, state, model, surro_model, adv_model, epsilon):
    action = model.predict(state)[0]
    #     print(action)
    effect = None
    attack = None
    _action = action
    _state = state
    state_range = np.array([epsilon])
    # print(state_range)
    op_action = get_op_action(env)
    #     op_action = (result.x)
    state = torch.from_numpy(state)

    state = state.detach().unsqueeze(0).requires_grad_(True)
    # print(model.predict(state, deterministic=True)[0])
    # next_origin_obs_dist = np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ _action - env.obstacle)
    # effect = next_origin_obs_dist
    effect = 1000
    for i in range(20):
        # print(state)
        action = model.policy._predict(state)[0].requires_grad_(True)
        action = torch.clip(action, torch.tensor([-1, -1]), torch.tensor([1, 1]))
        # print(action)

        action = action.double()


        # compute the distance
        loss = (torch.tensor([op_action]) - action).pow(2).sum().sqrt()

        # loss = mse(torch.tensor([op_action]).requires_grad_(True), action)
        model.policy.zero_grad()
        loss = loss.double()
        loss.backward()
        # print(loss)
        state_grad = state.grad.detach().squeeze()
        for i in range(15,18):
            if state_grad[i] > 0:
                state_grad[i] = 0
        perturbed_state = state - state_grad.sign() * epsilon * 0.1

        l = torch.from_numpy(_state) - torch.from_numpy(state_range)
        u = torch.from_numpy(_state) + torch.from_numpy(state_range)
        perturbed_state = torch.max(torch.min(perturbed_state, u), l)
        # scaled_action = model.policy._predict(perturbed_state)[0]
        # low, high = -1, 1
        # action_next = torch.clip(scaled_action, low, high)
        state = perturbed_state.detach().requires_grad_(True)

        # dist = np.linalg.norm(_state - env.center)
        # obs_dist = np.linalg.norm(_state - env.obstacle)
        # pertub_dist = np.linalg.norm(state.detach().numpy() - env.center)
        dist = 5 - (np.sqrt(np.sum(np.square(_state[15:18]))))
        pertub_dist = 5 - torch.sqrt(torch.square(state[0][15:18]).sum())
        # pertub_obs_dist = max(state[0][28:44].detach().numpy())
        # next_obs_dist = np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ action_next.detach().numpy() - env.obstacle)
        # print(dist -pertub_dist)
        if pertub_dist <= dist:
            #
            if loss <= effect:
                effect = loss
                # print(effect)
                attack = (state.detach().numpy() - _state)[0]
    # if pertub_obs_dist - 0.2 > 0:
    #
    #     if next_obs_dist < effect:
    #         effect = next_obs_dist
            # print(effect)
    # print(action, op_action)
    # print('+++++++++++++++++++')
    if attack is None:
        return np.zeros_like(_state)
    else:
        # print('find solution')
        return attack





def white(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon

    epoch = 0
    reach = 0
    violate = 0
    rho_list = []
    perturb_rho_list = []
    while True:
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        while True:
            attack = white_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            goal_dist = math.sqrt(obs[15]**2 + obs[16]**2 + obs[17]**2) - 4
            rho.append(goal_dist)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)
            goal_dist = math.sqrt(obs[15]**2 + obs[16]**2 + obs[17]**2) - 4

            perturb_rho.append(goal_dist)

            obs, reward, done, trun, info = env.step(action)
            x = env.agent.pos[0]
            # print(obs[0:12])
            # if done:
            #     epoch += 1
                # break
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                break
            if trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                if not done:
                    reach += 1
                epoch += 1

                break
        if epoch >= total_epoch:
            break
    print(f'white attack violation:{violate}, reach:{reach}')
    return rho_list, perturb_rho_list

def grey_c(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon
    rho_list = []
    perturb_rho_list = []
    epoch = 0
    reach = 0
    violate = 0
    while True:
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        while True:
            attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            goal_dist = math.sqrt(obs[15] ** 2 + obs[16] ** 2 + obs[17] ** 2) - 4
            rho.append(goal_dist)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)
            goal_dist = math.sqrt(obs[15] ** 2 + obs[16] ** 2 + obs[17] ** 2) - 4

            perturb_rho.append(goal_dist)

            obs, reward, done, trun, info = env.step(action)
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                break
            if trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                if not done:
                    reach += 1
                epoch += 1

                break
        if epoch >= total_epoch:
            break
    print(f'grey_c attack violation:{violate}, reach:{reach}')
    return rho_list, perturb_rho_list



def grey_s(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon

    rho_list = []
    perturb_rho_list = []
    epoch = 0
    reach = 0
    violate = 0
    while True:
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        while True:
            attack = white_attack(env, obs, model=surro_model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            goal_dist = math.sqrt(obs[15] ** 2 + obs[16] ** 2 + obs[17] ** 2) - 4
            rho.append(goal_dist)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)
            goal_dist = math.sqrt(obs[15] ** 2 + obs[16] ** 2 + obs[17] ** 2) - 4

            perturb_rho.append(goal_dist)

            obs, reward, done, trun, info = env.step(action)
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                break
            if trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                if not done:
                    reach += 1
                epoch += 1

                break
        if epoch >= total_epoch:
            break
    print(f'grey_s attack violation:{violate}, reach:{reach}')
    return rho_list, perturb_rho_list

def black(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon

    rho_list = []
    perturb_rho_list = []
    epoch = 0
    reach = 0
    violate = 0
    while True:
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        while True:
            attack = black_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            goal_dist = math.sqrt(obs[15] ** 2 + obs[16] ** 2 + obs[17] ** 2) - 4
            rho.append(goal_dist)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)
            goal_dist = math.sqrt(obs[15] ** 2 + obs[16] ** 2 + obs[17] ** 2) - 4

            perturb_rho.append(goal_dist)

            obs, reward, done, trun, info = env.step(action)
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                break
            if trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                if not done:
                    reach += 1
                epoch += 1

                break
        if epoch >= total_epoch:
            break
    print(f'black attack violation:{violate}, reach:{reach}')
    return rho_list, perturb_rho_list

def gradient(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon

    epoch = 0
    reach = 0
    violate = 0
    while True:
        obs, info = env.reset()

        while True:
            attack = Gradient_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            x = env.agent.pos[0]
            # print(obs[0:12])
            if info['cost'] != 0:
                violate += 1
                epoch += 1

                break
            if trun:
                if not done:
                    reach += 1
                epoch += 1
                break
        if epoch >= total_epoch:
            break
    print(f'GA violation:{violate}, reach:{reach}')
def laa(env, model, surro_model, adv_model, baseline_model,epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon

    epoch = 0
    reach = 0
    violate = 0
    while True:
        obs, info = env.reset()
        re = False
        while True:
            attack = laa_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, baseline_model=baseline_model, epsilon=epsilon)
            # print(attack)
            goal_dist = math.sqrt(obs[15] ** 2 + obs[16] ** 2 + obs[17] ** 2) - 2
            if goal_dist > 0:
                re = True
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            x = env.agent.pos[0]
            # print(obs[0:12])
            if info['cost'] != 0:
                violate += 1
                epoch += 1

                break
            if trun:
                if not done :
                    reach += 1
                epoch += 1
                break
        if epoch >= total_epoch:
            break
    print(f'laa violation:{violate}, reach:{reach}')
def mad(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon

    epoch = 0
    reach = 0
    violate = 0
    while True:
        obs, info = env.reset()

        while True:
            attack = MAD(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            x = env.agent.pos[0]
            # print(obs[0:12])
            if info['cost'] != 0:
                violate += 1
                epoch += 1

                break
            if trun:
                if not done:
                    reach += 1
                epoch += 1
                break
        if epoch >= total_epoch:
            break
    print(f'mad violation:{violate}, reach:{reach}')
env = SafetyCarCircle(render_mode='human')
from gymnasium.wrappers import TimeLimit
# env = CartPoleEnv()
env.env = TimeLimit(env.env, max_episode_steps=200)
model = PPO.load('train/model/SafetyCarCirclePPO-1.zip', env=env)
surro_model = SAC.load('train/model/surro_SafetyCarCircleSAC-1.zip', env=env)
# surro_model = PPO.load('train/model/SafetyCarCirclePPO-1.zip', env=env)
adv_model = PPO.load('train/model/adv_SafetyCarCircle-PPO-1.zip', env=env)
baseline_model = PPO.load('train/model/baseline_SafetyCarCircle1-PPO.zip', env=env)
total_epoch = 250
for epsilon in  [ 0.01, 0.05, 0.10, 0.15]:
# for epsilon in  [ 0.1]:
    print(epsilon)
    # rho_list, perturb_rho_list = black(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    # rho_list, perturb_rho_list = white(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,  total_epoch= total_epoch)
    # rho_list, perturb_rho_list  = grey_c(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon , total_epoch= total_epoch)
    # rho_list, perturb_rho_list  = grey_s(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    # gradient(env=env,  model=surro_model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    mad(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    # laa(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, baseline_model=baseline_model, total_epoch=total_epoch)

    print('++++++++++++')
# method = 'grey_c_'
# max_len = 0
# for i in range(0, len(rho_list)):
#     max_len = max(max_len, len(rho_list[i]))
# rho_ave = [0] * max_len
# pertub_rho_ave = [0] * max_len
# dist_ave = [0] * max_len
#
#
# for j in range(0, total_epoch):
#     temp1 = 0
#     temp2 = 0
#     for i in range(0, len(rho_list[j])):
#         rho_ave[i] =  rho_ave[i] + rho_list[j][i]/total_epoch
#         pertub_rho_ave[i] = pertub_rho_ave[i] + perturb_rho_list[j][i]/total_epoch
# # #
# with open(method + r'rho.txt', 'w') as fp:
#     for item in rho_ave:
#         # write each item on a new line
#         fp.write("%s\n" % item)
# with open(method + r'pertub_rho.txt', 'w') as fp:
#     for item in pertub_rho_ave:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#
#
#



# 0.01
# C:\Users\sjiang5\PycharmProjects\attack_nnController\safegym-attack\CarCircle\attack.py:34: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\torch\csrc\utils\tensor_new.cpp:248.)
#   loss = (torch.tensor([op_action]) - action).pow(2).sum().sqrt()
# black attack violation:34, reach:216
# white attack violation:45, reach:205
# grey_c attack violation:40, reach:210
# grey_s attack violation:32, reach:218
# C:\Users\sjiang5\PycharmProjects\attack_nnController\safegym-attack\CarCircle\baseline_attack.py:17: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
#   state = torch.tensor(state)
# GA violation:0, reach:250
# ++++++++++++
# 0.05
# black attack violation:37, reach:213
# white attack violation:148, reach:102
# grey_c attack violation:30, reach:220
# grey_s attack violation:35, reach:215
# GA violation:122, reach:128
# ++++++++++++
# 0.1
# black attack violation:36, reach:214
# white attack violation:208, reach:42
# grey_c attack violation:18, reach:232
# grey_s attack violation:50, reach:200
# GA violation:147, reach:103
# ++++++++++++
# 0.15
# black attack violation:26, reach:224
# white attack violation:226, reach:24
# grey_c attack violation:8, reach:242
# grey_s attack violation:38, reach:212
# GA violation:141, reach:109