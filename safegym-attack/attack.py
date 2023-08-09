import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, root

import torch
import torch.nn as nn
def black_attack(env, state, model, surro_model, adv_model, epsilon, policy, norm, args):
    action = surro_model.predict(state)[0]
    #     print(action)
    effect = None
    attack = None
    _action = action
    _state = state

    op_action = adv_model.predict(state)[0]
    #     op_action = (result.x)
    state = torch.from_numpy(state)

    state = state.detach().unsqueeze(0).requires_grad_(True)
    # print(model.predict(state, deterministic=True)[0])
    next_origin_obs_dist = np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ _action - env.obstacle)
    effect = next_origin_obs_dist
    for i in range(100):

        action = surro_model.policy._predict(state)[0].requires_grad_(True)
        #         print(action)
        #         action = action * 10
        action = action.double()
        mse = nn.MSELoss()

        # compute the distance
        pdist = torch.nn.PairwiseDistance(p=2)
        loss = pdist(torch.tensor([op_action]), action)

        # loss = mse(torch.tensor([op_action]).requires_grad_(True), action)
        surro_model.policy.zero_grad()
        loss = loss.double()
        # print(action.dtype)
        loss.backward()

        state_grad = state.grad.detach().squeeze()
        perturbed_state = state - epsilon / 30 * state_grad.sign()

        scaled_action = surro_model.policy._predict(perturbed_state)[0]
        low, high = -10, 10
        action_next = torch.clip(scaled_action, low, high)
        state = perturbed_state.detach().requires_grad_(True)

        dist = np.linalg.norm(_state - env.center)
        obs_dist = np.linalg.norm(_state - env.obstacle)
        pertub_dist = np.linalg.norm(state.detach().numpy() - env.center)
        pertub_obs_dist = np.linalg.norm(state.detach().numpy() - env.obstacle)
        next_obs_dist = np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ action_next.detach().numpy() - env.obstacle)
        if pertub_obs_dist - env.safe_norm_radius > 0:

            if next_obs_dist < effect:
                effect = next_obs_dist
                # print(effect)
                attack = (state.detach().numpy() - _state)[0]
    if attack is None:
        return np.zeros_like(_state)
    else:
        # print('find solution')
        return attack
