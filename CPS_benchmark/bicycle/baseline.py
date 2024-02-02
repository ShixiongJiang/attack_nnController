import torch
from scipy.optimize import Bounds, minimize

from CPS_benchmark.bicycle.attack_methods import black_attack, white_attack
import numpy as np

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

    result = minimize(fun, x_start, method='trust-constr', bounds=bounds)
    # print(result.x)

    op_action = (result.x)
    return op_action

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
        if Q_adv <= Q:
            Q = Q_adv
            attack = (state.detach().numpy() - _state)
    if attack is not None:
        # print(attack)
        return attack

    else:
        # print("no attack")
        return np.zeros_like(_state)
def gradient(env, model, surro_model, adv_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    dist_list = []
    for j in range(total_epoch):
        state = env.reset()
        dist_ = []
        for i in range(env.step_const):
            attack = Gradient_attack(env, state, model=model, surro_model=model, adv_model=adv_model, epsilon=epsilon)

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
    print(f'gradient attack violation:{number_violate}, reach:{num_reached}')
    return dist_list


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
            # pertub_action_list.append(pertub_action[0])
            state = new_state
            # action_list.append(action)

            # obs_dists.append(obs_dist)
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

def laa(env, model, surro_model, adv_model,laa_model, epsilon, total_epoch=100):
    epsilon = epsilon

    num_reached = 0
    number_violate = 0
    dist_list = []
    for j in range(total_epoch):
        state = env.reset()
        dist_ = []
        for i in range(2 * env.step_const):
            attack = black_attack(env=env, state=state, model=model, surro_model=surro_model, adv_model=laa_model, epsilon=epsilon)

            pertub_state = state + attack

            pertub_action = model.predict(pertub_state, deterministic=True)[0]

            new_state, reward, done, _ = env.step(pertub_action)

            dist = np.linalg.norm(state - env.center)
            obs_dist = min(np.linalg.norm(state - env.obstacle[0]), np.linalg.norm(state - env.obstacle[1]))
            dist_.append(obs_dist)
            # pertub_action_list.append(pertub_action[0])
            state = new_state
            # action_list.append(action)

            # obs_dists.append(obs_dist)
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
    print(f'laa attack violation:{number_violate}, reach:{num_reached}')
    return dist_list