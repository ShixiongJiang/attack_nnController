import math

import numpy as np

import torch
import torch.nn as nn
from customEnv.env import SafetyPointGoal1
from stable_baselines3 import PPO, SAC
from baseline import Gradient_attack, MAD


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
        #         print(action)
        #         action = action * 10
        action = action.double()

        # compute the distance
        pdist = torch.nn.PairwiseDistance(p=2)
        loss = (torch.tensor([op_action]) - action).pow(2).sum().sqrt()
        # loss = mse(torch.tensor([op_action]).requires_grad_(True), action)
        surro_model.policy.zero_grad()
        loss = loss.double()
        # print(action.dtype)
        loss.backward()
        # print(loss)
        state_grad = state.grad.detach().squeeze()
        # print(state_grad[torch.argmax(state_grad[12:28])])
        for i in range(12, 28):
            if state_grad[i] > 0:
                state_grad[i] = 0

        dist = 3 - max(_state[12:28]) * 3
        pertub_dist = 3 - max(state[0][12:28]) * 3

        if pertub_dist <= dist:
            #
            if loss <= effect:
                effect = loss
                attack = (state.detach().numpy() - _state)[0]

        perturbed_state = state - state_grad.sign() * epsilon * 0.1
        l = torch.from_numpy(_state) - torch.from_numpy(state_range)
        u = torch.from_numpy(_state) + torch.from_numpy(state_range)
        perturbed_state = torch.max(torch.min(perturbed_state, u), l)
        state = perturbed_state.detach().requires_grad_(True)

    if attack is None:
        return np.zeros_like(_state)
    else:
        # print('find solution')
        return attack


def get_op_action(obs):
    obstacle = np.argmax(obs[28:44])
    # print(obstacle)
    turn = obs[4]
    velo = obs[3]
    if obstacle > 4 and obstacle < 12:
        action0 = -1
    else:
        action0 = 1
    if obstacle > 4 and obstacle < 12:
        if obstacle < 8:
            action1 = -1 * (8 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
        elif obstacle > 8:
            action1 = 1 * (obstacle - 8) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
        else:
            action1 = 0
    else:
        if obstacle <= 4:
            action1 = 1 * (obstacle + 1) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
        elif obstacle >= 13 and obstacle < 16:
            action1 = -1 * (15 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
        else:
            action1 = 0
    action = [action0, action1]
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
    op_action = get_op_action(_state)
    #     op_action = (result.x)
    state = torch.from_numpy(state)

    state = state.detach().unsqueeze(0).requires_grad_(True)
    # print(model.predict(state, deterministic=True)[0])
    # next_origin_obs_dist = np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ _action - env.obstacle)
    # effect = next_origin_obs_dist
    effect = 1000
    for i in range(20):

        action = model.policy._predict(state)[0].requires_grad_(True)
        #         print(action)
        #         action = action * 10
        action = action.double()

        # compute the distance
        pdist = torch.nn.PairwiseDistance(p=2)
        loss = (torch.tensor([op_action]) - action).pow(2).sum().sqrt()

        # loss = mse(torch.tensor([op_action]).requires_grad_(True), action)
        model.policy.zero_grad()
        loss = loss.double()
        # print(action.dtype)
        loss.backward()
        # print(loss)
        state_grad = state.grad.detach().squeeze()
        # print(state_grad[torch.argmax(state_grad[12:28])])
        for i in range(12, 28):
            if state_grad[i] > 0:
                state_grad[i] = 0

        dist = 3 - max(_state[12:28]) * 3
        pertub_dist = 3 - max(state[0][12:28]) * 3

        if pertub_dist <= dist:
            #
            if loss <= effect:
                effect = loss
                # print(effect)
                attack = (state.detach().numpy() - _state)[0]

        perturbed_state = state - state_grad.sign() * epsilon * 0.1
        l = torch.from_numpy(_state) - torch.from_numpy(state_range)
        u = torch.from_numpy(_state) + torch.from_numpy(state_range)
        perturbed_state = torch.max(torch.min(perturbed_state, u), l)
        state = perturbed_state.detach().requires_grad_(True)

    if attack is None:
        return np.zeros_like(_state)
    else:
        # print('find solution')
        return attack


epsilon = 0.7


def white(env, model, surro_model, adv_model, epsilon, total_epoch):
    epoch = 0
    reach = 0
    violate = 0

    white_dis_list = []
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        dis = []


        while True:
            goal_dist = 3 - 3 * max(obs[12:28])
            obs_dist = 3 - 3 * max(obs[28:44])
            dis.append(obs_dist)
            rho.append(-goal_dist)
            attack = white_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            obs = attack + obs
            pertub_goal_dist = 3 - 3 * max(obs[12:28])
            perturb_rho.append(-pertub_goal_dist)

            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:

                epoch += 1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                white_dis_list.append(dis)
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:

                    violate += 1
                break
        if epoch >= total_epoch:
            break
    print(f'white attack violation:{violate}, reach:{reach}')
    return white_dis_list
def black(env,  model, surro_model, adv_model, epsilon, total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    black_dist_list = []
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        dist_list = []
        while True:
            attack = black_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)
            obs_dist = 3 - 3 * max(obs[28:44])
            dist_list.append(obs_dist)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:

                epoch +=1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                black_dist_list.append(dist_list)
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                # obs, info = env.reset()
                break
        if epoch >= total_epoch:
            break
    print(f'black attack violation:{violate}, reach:{ reach}')
    return black_dist_list
# env.close()
#
def gradient(env,  model, surro_model, adv_model, epsilon,  total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    gradient_dist_list = []
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        dist_list = []
        while True:
            attack = Gradient_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)

            obs = obs + attack

            action, _state = model.predict(obs, deterministic=True)
            obs_dist = 3 - 3 * max(obs[28:44])
            dist_list.append(obs_dist)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:

                epoch +=1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                gradient_dist_list.append(dist_list)
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                # obs, info = env.reset()
                break
        if epoch >= total_epoch:
            break
    print(f'Gradient attack violation:{violate}, reach:{ reach}')
    return gradient_dist_list

def mad(env,  model, surro_model, adv_model, epsilon,  total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        while True:
            attack = MAD(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            # print(attack - obs)
            obs = attack + obs

            action, _state = model.predict(obs, deterministic=True)


            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:

                epoch +=1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                # obs, info = env.reset()
                break
        if epoch >= total_epoch:
            break
    print(f'mad attack violation:{violate}, reach:{ reach}')
## grey box: without contorl policy
def grey_s(env,  model, surro_model, adv_model, epsilon, total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    grey_s_dist_list = []

    while True:
        obs, info = env.reset()
        dist_list = []
        while True:
            attack = white_attack(env, obs, model=surro_model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = obs + attack

            action, _state = model.predict(obs, deterministic=True)
            obs_dist = 3 - 3 * max(obs[28:44])
            dist_list.append(obs_dist)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:

                epoch += 1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                grey_s_dist_list.append(dist_list)
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                # obs, info = env.reset()
                break
        if epoch >= total_epoch:
            break
    print(f'Grey box without control policy violation:{violate}, reach:{ reach}')
    return grey_s_dist_list

## grey box: without sys
def grey_c(env,  model, surro_model, adv_model, epsilon, total_epoch=100):
    epoch = 0
    reach = 0
    violate = 0
    grey_c_dist_list = []

    while True:
        obs, info = env.reset()
        dist_list = []
        while True:
            attack = black_attack(env, obs, model=model, surro_model=model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = obs + attack

            action, _state = model.predict(obs, deterministic=True)
            obs_dist = 3 - 3 * max(obs[28:44])
            dist_list.append(obs_dist)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:

                epoch += 1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                grey_c_dist_list.append(dist_list)
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                # obs, info = env.reset()
                break
        if epoch >= total_epoch:
            break
    print(f'Grey box without sys violation:{violate}, reach:{ reach}')
    return grey_c_dist_list



env = SafetyPointGoal1()
model = PPO.load('train/model/SafetyPointGoal1-PPO-7.zip', env=env)
surro_model = SAC.load('train/model/surro_SafetyPointGoal1-SAC-1.zip', env=env)
adv_model = PPO.load('train/model/Adv_SafetyPointGoal1-PPO.zip', env=env)

obs, info = env.reset()
total_epoch = 100
env = SafetyPointGoal1(render_mode=None)
# gradient(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)

# for epsilon in [0.01, 0.05, 0.1, 0.15]:
for epsilon in [0.1]:
    print(epsilon)
    black_dist_list = black(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    white_dist_list = white(env=env, model=model, surro_model=surro_model, adv_model=adv_model,
                                                 epsilon=epsilon, total_epoch=total_epoch)

    grey_c_dist_list = grey_c(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon , total_epoch= total_epoch)
    grey_s_dist_list = grey_s(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    # gradient(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    # mad(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    print('++++++++++++')

import matplotlib.pyplot as plt
#
# max_len = 0
# for i in range(0, len(dist_list)):
#     max_len = max(max_len, len(dist_list[i]))
# rho_ave = [0] * max_len
# pertub_rho_ave = [0] * max_len
# dist_ave = [0] * max_len
# clean_dist_ave = [0] * max_len
# print(len(rho_ave))
# print((rho_list[0]))
# print(rho_list[1])


# for j in range(0, total_epoch):
#     temp1 = 0
#     temp2 = 0
#     for i in range(0, len(rho_list[j])):
#         rho_ave[i] =  rho_ave[i] + rho_list[j][i]/total_epoch
#         pertub_rho_ave[i] = pertub_rho_ave[i] + pertub_rho_list[j][i]/total_epoch
#

# rho_ave = rho_ave/total_epoch
# plt.plot( range(0, max_len), rho_ave )
# plt.plot( range(0, max_len), pertub_rho_ave)
# print(dist_list[1])
# for i in range(0, len(dist_list)):
#     for j in range(0, len(dist_list[i])):
#         dist_ave[j] = dist_ave[j] + dist_list[i][j] / len(dist_list)
# for i in range(0, len(clean_dist_list)):
#     for j in range(0, len(clean_dist_list[i])):
#         clean_dist_ave[j] = clean_dist_ave[j] + clean_dist_list[i][j] / len(clean_dist_list)
# # print(dist_ave)
# plt.plot(range(0, len(dist_ave)), dist_ave, label = "line 1")
# plt.plot(range(0, len(clean_dist_ave)), clean_dist_ave, label = 'without adversary')
# plt.ylim((-1, 8))
# plt.axhline(y=0.3, color='g', linestyle='--')
white_ave = []
black_ave = []
grey_c_ave = []
grey_s_ave = []
for i in range(0, len(white_dist_list)):
    white_ave.append(np.min(white_dist_list[i])  )
for i in range(0, len(black_dist_list)):
    black_ave.append(np.min(black_dist_list[i])  )
for i in range(0, len(grey_c_dist_list)):
    grey_c_ave.append(np.min(grey_s_dist_list[i]) )
for i in range(0, len(grey_s_dist_list)):
    grey_s_ave.append(np.min(grey_s_dist_list[i]) )



plt.yticks(fontsize=18)
plt.xticks(fontsize=18)

plt.legend()
plt.show()
plt.savefig('stealthy.png', dpi=500)
data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
plt.boxplot(data)
plt.axhline(y=0, color='g', linestyle='--')

plt.show()


with open(r'white_dist.txt', 'w') as fp:
    for item in white_ave:
        # write each item on a new line
        fp.write("%s\n" % item)
with open(r'black_dist.txt', 'w') as fp:
    for item in black_ave:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'grey_c_dist.txt', 'w') as fp:
    for item in grey_c_ave:
        # write each item on a new line
        fp.write("%s\n" % item)
with open(r'grey_s_dist.txt', 'w') as fp:
    for item in grey_s_ave:
        # write each item on a new line
        fp.write("%s\n" % item)
