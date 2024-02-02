import math

import numpy as np
import torch
from scipy.optimize import Bounds, NonlinearConstraint
from scipy.stats import beta
def Gradient_attack(env, state, model, surro_model, adv_model, epsilon):
    # print(type(state))
    _action = surro_model.predict(state)[0]
    _state = state
    action = surro_model.predict(state)[0]
    state_range = _state / np.linalg.norm(_state) * epsilon * math.sqrt(44)
    # print(surro_model.policy)
    Q = surro_model.critic_target(torch.from_numpy(_state).view(1,-1), torch.from_numpy(action).view(1,-1))[0]
    attack = None
    for i in range(20):
        state = torch.tensor(state)
        # print(state)
        state.requires_grad_(True)

        Q_cur = surro_model.critic_target(state.view(1,-1), torch.from_numpy(action).view(1, -1))[0]
        # print(Q_cur)
        model.policy.zero_grad()

        Q_cur.backward()
        grad = state.grad
        grad_dir = grad.sign() * epsilon * 0.1

        state = state - grad_dir

        state_range = np.array([epsilon])
        l = torch.from_numpy(_state) - torch.from_numpy(state_range)
        u = torch.from_numpy(_state) + torch.from_numpy(state_range)
        state = torch.max(torch.min(state, u), l)
        action = surro_model.predict(state.detach().numpy())[0]
        Q_adv = surro_model.critic(torch.from_numpy(_state).view(1, -1), torch.from_numpy(action).view(1, -1))[0]
        if Q_adv <= Q:
            Q = Q_adv
            attack = (state.detach().numpy() - _state)
    if attack is not None:
        # print(attack)
        return attack

    else:
        # print("no attack")
        return np.zeros_like(_state)

    # print(grad)
    # sign_data_grad = data_grad.sign()
    # perturbed_image = image + epsilon * sign_data_grad
    # # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # # Return the perturbed image
    # return perturbed_image
import torch.distributions.kl as kl
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, root

def MAD(env, obs, model, surro_model, adv_model, epsilon):
        action = model.predict(obs, deterministic=True)[0]
        # print(action)
        effect = None
        attack = None
        _action = action
        _state = obs
        state_range = np.array([epsilon])

        def fun(x):
            u = model.predict(x + _state, deterministic=True)[0]
            result = np.sum(np.power(action - u, 2)) * (-1)
            return result

        x_start = np.zeros_like(_state)
        lim = np.zeros_like(_state) + epsilon

        bounds = Bounds(-lim, lim)

        result = minimize(fun, x_start, method='trust-constr', bounds=bounds, tol=0.1)
        # print(result.x)

        op_action = (result.x)
        return op_action

def laa_attack(env, state, model, surro_model, adv_model,laa_model, epsilon):
    action = surro_model.predict(state)[0]
    #     print(action)
    effect = None
    attack = None
    _action = action
    _state = state
    state_range = np.array([epsilon])

    op_action = laa_model.predict(state)[0]
    #     op_action = (result.x)
    state = torch.from_numpy(state)

    state = state.detach().unsqueeze(0).requires_grad_(True)
    effect = 1000
    for i in range(20):

        action = surro_model.policy._predict(state)[0].requires_grad_(True)
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
        return attack


