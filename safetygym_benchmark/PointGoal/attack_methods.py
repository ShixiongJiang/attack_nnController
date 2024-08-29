import math

import numpy as np
import cvxpy as cp

import torch
from scipy.optimize import Bounds, minimize


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


def get_op_action(_state, _action, previous_u, A, B):
    def control_objective(u, x_current, A, B):
        # Predict the next state using the linear model
        x_next = A @ x_current + B @ u

        # Objective: maximize the maximum value in state[28:44]
        # Since most optimizers minimize, we return the negative of the objective
        return -np.max(x_next[28:44])

    # Initial guess for control input
    u_initial = np.zeros(2)  # Replace with an initial guess for the control input (2-dimensional)

    # Bounds for control inputs (if applicable)
    u_bounds = [(-1, 1), (-1, 1)]  # Replace with actual bounds

    # Perform the optimization to maximize the desired state value
    result = minimize(control_objective, u_initial, args=(_state, A, B), bounds=u_bounds, method='SLSQP')

    # Extract the optimal control input
    u_opt = result.x
    return u_opt

    # def fun(x, A, B, _state, previous_u, lambda_reg=0):
    #
    #     predicted_state = _state
    #     for _ in range(horizon):
    #         predicted_state = A @ predicted_state + B @ x
    #
    #     max_value = np.max(predicted_state[28:44])
    #     regularization = lambda_reg * np.sum((x - previous_u)**2)
    #     # Return the negative of the max value
    #     return -max_value + regularization
    #
    # # Define horizon
    # horizon = 1  # You can extend this
    # lim = [-1, 1]
    # bounds = Bounds(lim[0], lim[1])
    #
    # result = minimize(fun, _action, args=(A, B, _state, previous_u), method='Powell', bounds=bounds)
    # op_action = result.x
    #
    # return op_action

# def get_op_action(obs):

    # obstacle = np.argmax(obs[28:44])
    #
    # turn = obs[4]
    # velo = obs[3]
    # if obstacle > 4 and obstacle < 12:
    #     action0 = -1
    # else:
    #     action0 = 1
    # if obstacle > 4 and obstacle < 12:
    #     if obstacle < 8:
    #         action1 = -1 * (8 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
    #     elif obstacle > 8:
    #         action1 = 1 * (obstacle - 8) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
    #     else:
    #         action1 = 0
    # else:
    #     if obstacle <= 4:
    #         action1 = 1 * (obstacle + 1) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
    #     elif obstacle >= 13 and obstacle < 16:
    #         action1 = -1 * (15 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(obs[28:44]))
    #     else:
    #         action1 = 0
    # action = [action0, action1]
    # return action
def white_attack(env, state, model, surro_model, adv_model, epsilon):
    action = model.predict(state)[0]
    #     print(action)
    effect = None
    attack = None
    _action = action
    _state = state
    state_range = np.array([epsilon])
    # print(state_range)
    # op_action = get_op_action(_state)
    op_action = get_op_action(_state=_state, _action=_action, A=env.A, B=env.B)
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
        for i in range(12,28):
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
def white(env,  model, surro_model, adv_model, epsilon, total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    rho_list = []
    perturb_rho_list = []
    dis_list = []
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        _state = obs
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
    print(f'white attack violation:{violate}, reach:{ reach}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return rho_list, perturb_rho_list, dis_list
# env.close()
# env = SafetyPointGoal1(render_mode='rgb_array')
# obs, info = env.reset()

def black(env,  model, surro_model, adv_model, epsilon, total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    rho_list = []
    dis_list = []
    perturb_rho_list = []
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        rho = []
        perturb_rho = []
        obs, info = env.reset()
        dis = []
        while True:
            goal_dist = 3 - 3 * max(obs[12:28])
            obs_dist = 3 - 3 * max(obs[28:44])
            rho.append(-goal_dist)
            dis.append(obs_dist)
            attack = black_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = attack + obs
            pertub_goal_dist = 3 - 3 * max(obs[12:28])
            perturb_rho.append(-pertub_goal_dist)

            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                dis_list.append(dis)
                epoch += 1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                break
        if epoch >= total_epoch:
            break
    print(f'black attack violation:{violate}, reach:{ reach}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return rho_list, perturb_rho_list, dis_list


#

def grey_s(env,  model, surro_model, adv_model, epsilon, total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    rho_list = []
    dist_list = []
    perturb_rho_list = []
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        rho = []
        perturb_rho = []
        obs, info = env.reset()
        dist = []
        while True:
            goal_dist = 3 - 3 * max(obs[12:28])
            obs_dist = 3 - 3 * max(obs[28:44])
            rho.append(-goal_dist)
            dist.append(obs_dist)
            attack = white_attack(env, obs, model=surro_model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            obs = attack + obs
            pertub_goal_dist = 3 - 3 * max(obs[12:28])
            perturb_rho.append(-pertub_goal_dist)

            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                dist_list.append(dist)
                epoch += 1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                break
        if epoch >= total_epoch:
            break
    print(f'Grey box without control policy violation:{violate}, reach:{ reach}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return rho_list, perturb_rho_list, dist_list


## grey box: without sys
def grey_c(env,  model, surro_model, adv_model, epsilon, total_epoch=100):
    epoch = 0
    reach = 0
    violate = 0
    rho_list = []
    perturb_rho_list = []
    dist_list = []
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        rho = []
        perturb_rho = []
        obs, info = env.reset()
        dist = []
        while True:
            goal_dist = 3 - 3 * max(obs[12:28])
            obs_dist = 3 - 3 * max(obs[28:44])
            rho.append(-goal_dist)
            dist.append(obs_dist)
            attack = black_attack(env, obs, model=model, surro_model=model, adv_model=adv_model, epsilon=epsilon)
            obs = attack + obs
            pertub_goal_dist = 3 - 3 * max(obs[12:28])
            perturb_rho.append(-pertub_goal_dist)

            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:
                rho_list.append(rho)
                perturb_rho_list.append(perturb_rho)
                dist_list.append(dist)
                epoch += 1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                break
        if epoch >= total_epoch:
            break
    print(f'Grey box without sys violation:{violate}, reach:{ reach}, violation prob:{violate / total_epoch}, reach prob:{reach / total_epoch}')
    return rho_list, perturb_rho_list, dist_list


