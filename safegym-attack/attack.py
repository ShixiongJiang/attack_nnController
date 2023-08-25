import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, root

import torch
import torch.nn as nn
from customEnv.env import SafetyPointGoal1
from stable_baselines3 import A2C, PPO
def black_attack(env, state, model, surro_model, adv_model, epsilon):
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
    # next_origin_obs_dist = np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ _action - env.obstacle)
    # effect = next_origin_obs_dist
    for i in range(10):

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
        perturbed_state = state - epsilon / 20 * state_grad.sign()
        perturbed_state = torch.clip(perturbed_state, perturbed_state - 0.2, perturbed_state+0.2)
        scaled_action = surro_model.policy._predict(perturbed_state)[0]
        low, high = 0, 1
        action_next = torch.clip(scaled_action, low, high)
        state = perturbed_state.detach().requires_grad_(True)

        # dist = np.linalg.norm(_state - env.center)
        # obs_dist = np.linalg.norm(_state - env.obstacle)
        # pertub_dist = np.linalg.norm(state.detach().numpy() - env.center)
        pertub_obs_dist = max(state[0][28:44].detach().numpy())
        # next_obs_dist = np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ action_next.detach().numpy() - env.obstacle)
        attack = (state.detach().numpy() - _state)[0]

        # if pertub_obs_dist - 0.2 > 0:
        #
        #     if next_obs_dist < effect:
        #         effect = next_obs_dist
                # print(effect)

    if attack is None:
        return np.zeros_like(_state)
    else:
        # print('find solution')
        return attack

env = SafetyPointGoal1()
model = PPO.load('train/SafetyPointGoal1-PPO.zip', env=env)
adv_model = PPO.load('train/Adv_SafetyPointGoal1-PPO.zip', env=env)

obs, info = env.reset()

i = 0
reach = 0

for i in range(10000):
    attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
    # print(attack)
    obs = attack + obs
    action, _state = model.predict(obs, deterministic=True)


    obs, reward, done, trun, info = env.step(action)
    # print(obs[0:12])
    if done:
        goal_dist = 3 - 3 * max(obs[12:28])
        print(goal_dist)