import math

import torch
import numpy as np


from attack_methods import black_attack, white_attack
def Gradient_attack(env, state, model, surro_model, adv_model, epsilon):
    # print(type(state))
    _action = surro_model.predict(state)[0]
    _state = state
    action = surro_model.predict(state)[0]
    state_range = _state / np.linalg.norm(_state) * epsilon * math.sqrt(44)
    # print(surro_model.policy)
    Q = surro_model.critic_target(torch.from_numpy(_state).view(1, -1), torch.from_numpy(action).view(1, -1))[0]
    attack = None
    for i in range(20):
        state = torch.tensor(state)
        # print(state)
        state.requires_grad_(True)

        Q_cur = surro_model.critic_target(state.view(1, -1), torch.from_numpy(action).view(1, -1))[0]
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
        dist = np.linalg.norm(_state - env.center)
        pertub_dist = np.linalg.norm(state.detach().numpy() - env.center)
        if Q_adv <= Q and pertub_dist <= dist:
            Q = Q_adv
            attack = (state.detach().numpy() - _state)
    if attack is not None:
        # print(attack)
        return attack

    else:
        # print("no attack")
        return np.zeros_like(_state)


import torch.distributions.kl as kl
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, root


def MAD(env, obs, model, surro_model, adv_model, epsilon):
    _state = obs
    state_range = np.array([epsilon])
    dis = model.policy.get_distribution(torch.from_numpy(obs).view(1, -1))
    p = dis.distribution

    # print(p.loc)
    def fun(x):
        dis2 = model.policy.get_distribution(torch.from_numpy(obs + x).view(1, -1))
        # print(dis2)
        q = dis2.distribution
        # print(q)
        return -kl.kl_divergence(p, q).sum().detach().numpy()

    x_start = np.zeros_like(obs)
    # non_linear_eq = lambda x: (np.max(np.abs(x))) - epsilon
    bo = np.zeros_like(_state) + epsilon
    bound = Bounds(-bo, bo)
    result = minimize(fun, x_start, method='trust-constr', bounds=bound)
    op_attack = (result.x)
    # print(op_attack)
    return op_attack

def mad(env, model, surro_model, adv_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    dist_list = []
    for j in range(total_epoch):
        state = env.reset()
        # clean_state = clean_env.reset()
        dist_ = []
        for i in range(2 * env.step_const):
            attack = MAD(env, state, model, surro_model, adv_model, epsilon)

            pertub_state = state + attack

            pertub_action = model.predict(pertub_state, deterministic=True)[0]

            new_state, reward, done, _ = env.step(pertub_action)

            dist = np.linalg.norm(state - env.center)
            obs_dist = min(np.linalg.norm(state - env.obstacle[0]), np.linalg.norm(state - env.obstacle[1]))
            dist_.append(obs_dist)
            state = new_state
            if obs_dist <= env.safe_norm_radius:
                number_violate += 1
                dist_list.append(dist_)
                break
            if dist <= env.target_norm_radius:  # stop
                num_reached += 1
                dist_list.append(dist_)
                break
            if done:
                env.reset()
    print(f'mad attack violation:{number_violate}, reach:{num_reached}')
    return dist_list

def laa(env, model, surro_model, adv_model, laa_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    grey_c_dist_list = []
    for j in range(total_epoch):
        state = env.reset()
        dist_list = []
        for i in range(2 * env.step_const):
            attack = black_attack(env, state, model=model, surro_model=surro_model, adv_model=laa_model,
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
                grey_c_dist_list.append(dist_list)
                break
            if dist <= env.target_norm_radius:  # stop
                num_reached += 1
                grey_c_dist_list.append(dist_list)
                break
            if done:
                env.reset()
    print(f'laa attack violation:{number_violate}, reach:{num_reached}')




def gradient(env, model, surro_model, adv_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    for j in range(total_epoch):
        state = env.reset()
        for i in range(2 * env.step_const):
            attack = Gradient_attack(env, state, model=model, surro_model=model, adv_model=adv_model, epsilon=epsilon)

            pertub_state = state + attack

            pertub_action = model.predict(pertub_state, deterministic=True)[0]

            new_state, reward, done, _ = env.step(pertub_action)

            dist = np.linalg.norm(state - env.center)
            obs_dist = min(np.linalg.norm(state - env.obstacle[0]), np.linalg.norm(state - env.obstacle[1]))
            state = new_state
            # pertub_action_list.append(pertub_action[0])
            # action_list.append(action)

            # obs_dists.append(obs_dist)
            if obs_dist <= env.safe_norm_radius:
                number_violate += 1
                break
            if dist <= env.target_norm_radius:  # stop
                num_reached += 1
                break
            if done:
                env.reset()
    print(f'GA attack violation:{number_violate}, reach:{num_reached}')





