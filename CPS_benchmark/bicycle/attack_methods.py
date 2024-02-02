import torch
import math
from typing import Optional

import gym
import numpy as np
from gym import spaces
from scipy.integrate import odeint
from bicycle_env import bicycle
from scipy.optimize import minimize, Bounds


def white_attack(env, state, model, surro_model, adv_model, epsilon):
    action = model.predict(state, deterministic=True)[0]
    # print(action)
    effect = None
    attack = None
    _action = action
    _state = state
    state_range = np.array([epsilon])

    def fun(x):
        return min(np.linalg.norm(odeint(bicycle, state, ts, args=(x,))[-1] - env.obstacle[0]),
                   np.linalg.norm(odeint(bicycle, state, ts, args=(x,))[-1] - env.obstacle[1]))

    x_start = np.array(action)
    lim = [-7, 7]

    bounds = Bounds([lim[0], lim[0]], [lim[1], lim[1]])

    ts = [env.horizon[env.steps], env.horizon[env.steps + 1]]
    x_state = odeint(bicycle, state, ts, args=(action,))[-1]

    result = minimize(fun, x_start, method='trust-constr', bounds=bounds)
    # print(result.x)
    op_action = (result.x)
    state = torch.from_numpy(state)

    state = state.detach().unsqueeze(0).requires_grad_(True)

    effect = 1000
    for i in range(40):

        action = model.policy._predict(state, deterministic=True)[0].requires_grad_(True)
        action = action * 7
        action = action.double()
        #         mse = nn.MSELoss()

        # compute the distance
        loss = (torch.tensor([op_action]) - action).pow(2).sum().sqrt()
        # print(loss)
        # loss = mse(torch.tensor([op_action]).requires_grad_(True), action)
        model.actor.zero_grad()
        loss = loss.double()
        # print(action.dtype)
        loss.backward()
        state_grad = state.grad.detach().squeeze()
        perturbed_state = state - state_grad.sign() * epsilon * 0.1
        l = torch.from_numpy(_state) - torch.from_numpy(state_range)
        u = torch.from_numpy(_state) + torch.from_numpy(state_range)
        perturbed_state = torch.max(torch.min(perturbed_state, u), l)

        dist = np.linalg.norm(_state - env.center)
        pertub_dist = np.linalg.norm(state.detach().numpy() - env.center)

        if pertub_dist <= dist:
            # 
            if loss <= effect:
                effect = loss
                # print(effect)
                attack = (state.detach().numpy() - _state)[0]
        state = perturbed_state.detach().requires_grad_(True)

    if attack is None:
        return np.zeros_like(_state)
    else:
        # print('find solution')
        return attack


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

    for i in range(40):

        action = surro_model.policy._predict(state, deterministic=True)[0].requires_grad_(True)
        action = action * 7
        action = action.double().requires_grad_(True)
        # print(action)
        # compute the distance
        loss = (torch.tensor([op_action]) - action).pow(2).sum().sqrt()

        # loss = mse(torch.tensor([op_action]).requires_grad_(True), action)
        surro_model.policy.zero_grad()
        loss = loss.double()
        # print(action.dtype)
        loss.backward()
        state_grad = state.grad.detach().squeeze()
        # print(state_grad)
        perturbed_state = state - state_grad.sign() * epsilon * 0.1
        l = torch.from_numpy(_state) - torch.from_numpy(state_range)
        u = torch.from_numpy(_state) + torch.from_numpy(state_range)
        perturbed_state = torch.max(torch.min(perturbed_state, u), l)
        dist = np.linalg.norm(_state - env.center)
        pertub_dist = np.linalg.norm(state.detach().numpy() - env.center)

        if pertub_dist <= dist:
            #
            if loss <= effect:
                effect = loss
                # print(effect)
                attack = (state.detach().numpy() - _state)[0]
        state = perturbed_state.detach().requires_grad_(True)

    if attack is None:
        return np.zeros_like(_state)
    else:
        # print(attack)
        return attack



def white(env, model, surro_model, adv_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    white_dist_list = []
    for j in range(total_epoch):
        state = env.reset()
        dist_list = []
        # clean_state = clean_env.reset()
        for i in range(2 * env.step_const):
            attack = white_attack(env, state, model, surro_model, adv_model, epsilon)

            pertub_state = state + attack

            pertub_action = model.predict(pertub_state, deterministic=True)[0]

            new_state, reward, done, _ = env.step(pertub_action)

            dist = np.linalg.norm(state - env.center)

            obs_dist = min(np.linalg.norm(state - env.obstacle[0]), np.linalg.norm(state - env.obstacle[1]))
            dist_list.append(obs_dist)
            state = new_state
            # pertub_action_list.append(pertub_action[0])
            # action_list.append(action)

            # obs_dists.append(obs_dist)
            if obs_dist <= env.safe_norm_radius:
                number_violate += 1
                white_dist_list.append(dist_list)
                break
            if dist <= env.target_norm_radius:  # stop
                num_reached += 1
                white_dist_list.append(dist_list)
                break
            if done:
                env.reset()
    print(f'white attack violation:{number_violate}, reach:{num_reached}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return white_dist_list


def black(env, model, surro_model, adv_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    black_dist_list = []
    for j in range(total_epoch):
        state = env.reset()
        # clean_state = clean_env.reset()
        dist_list = []
        for i in range(2 * env.step_const):
            attack = black_attack(env, state, model, surro_model, adv_model, epsilon)

            pertub_state = state + attack

            pertub_action = model.predict(pertub_state, deterministic=True)[0]

            new_state, reward, done, _ = env.step(pertub_action)

            dist = np.linalg.norm(state - env.center)
            obs_dist = min(np.linalg.norm(state - env.obstacle[0]), np.linalg.norm(state - env.obstacle[1]))
            dist_list.append(obs_dist)
            # pertub_action_list.append(pertub_action[0])
            state = new_state
            # action_list.append(action)

            # obs_dists.append(obs_dist)
            if obs_dist <= env.safe_norm_radius:
                number_violate += 1
                black_dist_list.append(dist_list)
                break
            if dist <= env.target_norm_radius:  # stop
                num_reached += 1
                black_dist_list.append(dist_list)
                break
            if done:
                env.reset()
    print(f'black attack violation:{number_violate}, reach:{num_reached}, violation prob:{number_violate / total_epoch}, reach prob:{num_reached / total_epoch}')
    return black_dist_list


def grey_s(env, model, surro_model, adv_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    grey_s_dist_list = []
    for j in range(total_epoch):
        state = env.reset()
        # clean_state = clean_env.reset()
        dist_list = []
        for i in range(2 * env.step_const):
            attack = white_attack(env, state, model=surro_model, surro_model=surro_model, adv_model=adv_model,
                                  epsilon=epsilon)

            pertub_state = state + attack

            pertub_action = model.predict(pertub_state, deterministic=True)[0]

            new_state, reward, done, _ = env.step(pertub_action)

            dist = np.linalg.norm(state - env.center)
            obs_dist = min(np.linalg.norm(state - env.obstacle[0]), np.linalg.norm(state - env.obstacle[1]))
            dist_list.append(obs_dist)
            state = new_state
            # pertub_action_list.append(pertub_action[0])
            # action_list.append(action)

            # obs_dists.append(obs_dist)
            if obs_dist <= env.safe_norm_radius:
                number_violate += 1
                grey_s_dist_list.append(dist_list)
                break
            if dist <= env.target_norm_radius:  # stop
                num_reached += 1
                grey_s_dist_list.append(dist_list)
                break
            if done:
                env.reset()
    print(f'grey_s attack violation:{number_violate}, reach:{num_reached}, violation prob:{number_violate / total_epoch}, reach prob:{num_reached / total_epoch}')
    return grey_s_dist_list





def grey_c(env, model, surro_model, adv_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    grey_c_dist_list = []
    for j in range(total_epoch):
        state = env.reset()
        dist_list = []
        for i in range(2 * env.step_const):
            attack = black_attack(env, state, model=model, surro_model=model, adv_model=adv_model, epsilon=epsilon)

            pertub_state = state + attack

            pertub_action = model.predict(pertub_state, deterministic=True)[0]

            new_state, reward, done, _ = env.step(pertub_action)

            dist = np.linalg.norm(state - env.center)
            obs_dist = min(np.linalg.norm(state - env.obstacle[0]), np.linalg.norm(state - env.obstacle[1]))
            dist_list.append(obs_dist)
            state = new_state
            # pertub_action_list.append(pertub_action[0])
            # action_list.append(action)

            # obs_dists.append(obs_dist)
            if obs_dist <= env.safe_norm_radius:
                number_violate += 1
                grey_c_dist_list.append(dist_list)
                break
            if dist <= env.target_norm_radius:  # stop
                num_reached += 1
                grey_c_dist_list.append(dist_list)
                break
            if done:
                env.reset()
    print(f'grey_c attack violation:{number_violate}, reach:{num_reached}, violation prob:{number_violate / total_epoch}, reach prob:{num_reached / total_epoch}')
    return grey_c_dist_list
