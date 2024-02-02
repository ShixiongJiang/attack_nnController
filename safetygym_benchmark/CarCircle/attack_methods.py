import math

import numpy as np

import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC


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

def get_op_action(env):

    x = env.agent.pos[0]
    y = env.agent.pos[1]
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


epsilon = 0.7
def white(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon
    epoch = 0
    reach = 0
    violate = 0
    rho_list = []
    perturb_rho_list = []
    dist_list = []
    while True:
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        dist = []
        while True:
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis ** 2 + y_dis ** 2)
            dist.append(obs_dist)
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

            if info['cost'] != 0:
                violate += 1
                epoch += 1
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                dist_list.append(dist)
                break
            if trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                dist_list.append(dist)
                if not done:
                    reach += 1
                epoch += 1

                break
        if epoch >= total_epoch:
            break
    print(f'white attack violation:{violate}, reach:{reach}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return rho_list, perturb_rho_list, dist_list

def grey_c(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon
    rho_list = []
    perturb_rho_list = []
    epoch = 0
    reach = 0
    violate = 0
    dis_list = []
    while True:
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        dis = []
        while True:
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis ** 2 + y_dis ** 2)
            dis.append(obs_dist)
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
                dis_list.append(dis)
                break
            if trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                dis_list.append(dis)
                if not done:
                    reach += 1
                epoch += 1

                break
        if epoch >= total_epoch:
            break
    print(f'grey_c attack violation:{violate}, reach:{reach}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return rho_list, perturb_rho_list, dis_list



def grey_s(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon

    rho_list = []
    perturb_rho_list = []
    epoch = 0
    reach = 0
    violate = 0
    dis_list = []
    while True:
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        dis = []
        while True:
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis ** 2 + y_dis ** 2)
            dis.append(obs_dist)
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
                dis_list.append(dis)
                break
            if trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                dis_list.append(dis)
                if not done:
                    reach += 1
                epoch += 1

                break
        if epoch >= total_epoch:
            break
    print(f'grey_s attack violation:{violate}, reach:{reach}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return rho_list, perturb_rho_list, dis_list

def black(env, model, surro_model, adv_model, epsilon, total_epoch):
    obs, info = env.reset()
    epsilon = epsilon

    rho_list = []
    perturb_rho_list = []
    epoch = 0
    reach = 0
    violate = 0
    dist_list = []
    while True:
        obs, info = env.reset()
        rho = []
        perturb_rho = []
        dis = []
        while True:
            x = env.agent.pos[0]
            y = env.agent.pos[1]
            x_dis = min(abs(x - 1.125), abs(x + 1.25))
            if abs(y) < 0.3:
                obs_dist = x_dis
            else:
                y_dis = min(abs(y - 0.3), abs(y + 0.3))
                obs_dist = math.sqrt(x_dis ** 2 + y_dis ** 2)
            dis.append(obs_dist)
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
                dist_list.append(dis)
                break
            if trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                dist_list.append(dis)
                if not done:
                    reach += 1
                epoch += 1

                break
        if epoch >= total_epoch:
            break
    print(f'black attack violation:{violate}, reach:{reach}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return rho_list, perturb_rho_list, dist_list


