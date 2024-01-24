import math

import numpy as np

import torch
import torch.nn as nn
from train.customerEnv import SafetyCarCircle
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
    epoch = 0
    reach = 0
    violate = 0

    white_dis_list = []
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        dist_list = []


        while True:
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis **2 + y_dis **2)

            dist_list.append(obs_dist)
            attack = white_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            obs = attack + obs
            pertub_goal_dist = 3 - 3 * max(obs[12:28])
            perturb_rho.append(-pertub_goal_dist)

            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                white_dis_list.append(dist_list)
                break
            if trun:
                if not done:
                    reach += 1
                epoch += 1
                white_dis_list.append(dist_list)

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
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis ** 2 + y_dis ** 2)
            dist_list.append(obs_dist)

            attack = black_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)


            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            # if done or trun:
            #     # epoch += 1
            #     # break
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                black_dist_list.append(dist_list)
                break
            if trun:
                if not done:
                    reach += 1
                epoch += 1
                black_dist_list.append(dist_list)

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
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis ** 2 + y_dis ** 2)
            dist_list.append(obs_dist)
            attack = Gradient_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)

            obs = obs + attack

            action, _state = model.predict(obs, deterministic=True)
            obs_dist = 3 - 3 * max(obs[28:44])
            dist_list.append(obs_dist)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                gradient_dist_list.append(dist_list)
                break
            if trun:
                if not done:
                    reach += 1
                epoch += 1
                gradient_dist_list.append(dist_list)

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
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis ** 2 + y_dis ** 2)
            dist_list.append(obs_dist)
            attack = white_attack(env, obs, model=surro_model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = obs + attack

            action, _state = model.predict(obs, deterministic=True)
            # dist_list.append(obs_dist)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                grey_s_dist_list.append(dist_list)
                break
            if trun:
                if not done:
                    reach += 1
                epoch += 1
                grey_s_dist_list.append(dist_list)

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
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis ** 2 + y_dis ** 2)
            dist_list.append(obs_dist)
            attack = black_attack(env, obs, model=model, surro_model=model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = obs + attack

            action, _state = model.predict(obs, deterministic=True)
            obs_dist = 3 - 3 * max(obs[28:44])
            dist_list.append(obs_dist)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if info['cost'] != 0:
                violate += 1
                epoch += 1
                grey_c_dist_list.append(dist_list)
                break
            if trun:
                if not done:
                    reach += 1
                epoch += 1
                grey_c_dist_list.append(dist_list)

                break
        if epoch >= total_epoch:
            break
    print(f'Grey box without sys violation:{violate}, reach:{ reach}')
    return grey_c_dist_list



env = SafetyCarCircle()
model = PPO.load('train/model/SafetyCarCirclePPO-1.zip', env=env)
surro_model = SAC.load('train/model/surro_SafetyCarCircleSAC-1.zip', env=env)
# surro_model = PPO.load('train/model/SafetyCarCirclePPO-1.zip', env=env)
adv_model = PPO.load('train/model/adv_SafetyCarCircle-PPO-1.zip', env=env)

obs, info = env.reset()
total_epoch = 100
env = SafetyCarCircle(render_mode=None)
# gradient(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)

# for epsilon in [0.01, 0.05, 0.1, 0.15]:
for epsilon in [0.15]:
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
    white_ave.append(np.sum(white_dist_list[i])/len(white_dist_list[i])  )
for i in range(0, len(black_dist_list)):
    black_ave.append(np.sum(black_dist_list[i])/len(black_dist_list[i])  )
for i in range(0, len(grey_c_dist_list)):
    grey_c_ave.append(np.sum(grey_c_dist_list[i])/len(grey_c_dist_list[i]) )
for i in range(0, len(grey_s_dist_list)):
    grey_s_ave.append(np.sum(grey_s_dist_list[i])/len(grey_s_dist_list[i]) )
# for i in range(0, len(white_dist_list)):
#     white_ave.append(np.min(white_dist_list[i])  )
# for i in range(0, len(black_dist_list)):
#     black_ave.append(np.min(black_dist_list[i]))
# for i in range(0, len(grey_c_dist_list)):
#     grey_c_ave.append(np.min(grey_c_dist_list[i]) )
# for i in range(0, len(grey_s_dist_list)):
#     grey_s_ave.append(np.min(grey_s_dist_list[i]))



plt.yticks(fontsize=18)
plt.xticks(fontsize=18)

plt.legend()
plt.show()
plt.savefig('stealthy.png', dpi=500)
data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
plt.boxplot(data)
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
