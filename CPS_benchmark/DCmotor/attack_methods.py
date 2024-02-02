import numpy as np
import torch
from scipy.optimize import Bounds, minimize


def white_attack(env, state, model, surro_model, adv_model, epsilon):
    action = model.predict(state, deterministic=True)[0]
    # print(action)
    effect = None
    attack = None
    _action = action
    _state = state
    state_range = np.array([epsilon])

    def fun(x):
        return min(np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ x - env.obstacle[0]),
                   np.linalg.norm(env.sysd.A @ _state + env.sysd.B @ x - env.obstacle[1]))

    x_start = np.array(action)
    lim = [-10, 10]

    bounds = Bounds(lim[0], lim[1])

    #     x_state = odeint(bicycle, state, ts, args=(action,))[-1]
    #     non_linear_eq= lambda x: np.linalg.norm(odeint(bicycle, state, ts, args=(x,))[-1] - env.obstacle) -\
    #                             np.linalg.norm(odeint(bicycle, state, ts, args=(action,))[-1] - env.obstacle)
    #     non_linear_constr = NonlinearConstraint(non_linear_eq, 0.0, np.inf)
    #     result = minimize(fun, x_start, method='trust-constr', bounds=bounds, constraints=non_linear_constr)
    result = minimize(fun, x_start, method='trust-constr', bounds=bounds)
    # print(result.x)
    op_action = (result.x)
    state = torch.from_numpy(state)

    state = state.detach().unsqueeze(0).requires_grad_(True)
    # print(model.predict(state, deterministic=True)[0])
    # next_origin_obs_dist = min(np.linalg.norm( env.sysd.A @ _state + env.sysd.B @ _action - env.obstacle[0]),
    #                            np.linalg.norm( env.sysd.A @ _state + env.sysd.B @ _action - env.obstacle[1]))
    effect = 1000

    for i in range(20):

        action = model.policy._predict(state, deterministic=True)[0].requires_grad_(True)
        action = action * 10
        action = action.double().requires_grad_(True)
        # print(action)
        # compute the distance
        loss = (torch.tensor([op_action]) - action).pow(2).sqrt()

        # loss = mse(torch.tensor([op_action]).requires_grad_(True), action)
        model.policy.zero_grad()
        loss = loss.double()
        # print(action.dtype)
        loss.backward()

        state_grad = state.grad.detach().squeeze()
        # print(loss)
        perturbed_state = state - state_grad.sign() * epsilon * 0.1
        # print(perturbed_state)
        l = torch.from_numpy(_state) - torch.from_numpy(state_range)
        u = torch.from_numpy(_state) + torch.from_numpy(state_range)
        perturbed_state = torch.max(torch.min(perturbed_state, u), l)
        # print(perturbed_state)
        # print('++++++++++')
        # print(action_next.detach().numpy()
        state = perturbed_state.detach().requires_grad_(True)
        # print(f'state:{perturbed_state}')
        # print(f'action:{_action}, op_action:{op_action}')
        dist = np.linalg.norm(_state - env.center)
        obs_dist = min(np.linalg.norm(_state - env.obstacle[0]), np.linalg.norm(_state - env.obstacle[1]))
        pertub_dist = np.linalg.norm(state.detach().numpy() - env.center)
        # pertub_obs_dist = np.linalg.norm(state.detach().numpy() - env.obstacle)
        #         print(action_next)
        #         next_obs_dist =  np.linalg.norm(odeint(bicycle, state.detach()[0], ts, args=(action_next.detach(),))[-1] - env.obstacle)
        if pertub_dist <= dist:
            #
            if loss <= effect:
                effect = loss
                # print(effect)
                attack = (state.detach().numpy() - _state)[0]
    # print('+++++++++')
    if attack is None:
        return np.zeros_like(_state)
    else:
        # print(attack)
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

    for i in range(20):

        action = surro_model.policy._predict(state, deterministic=True)[0].requires_grad_(True)
        action = action * 10
        action = action.double().requires_grad_(True)
        # print(action)
        # compute the distance
        loss = (torch.tensor([op_action]) - action).pow(2).sqrt()

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

        # print(action_next.detach().numpy()
        state = perturbed_state.detach().requires_grad_(True)
        # print(f'state:{perturbed_state}')
        # print(f'action:{_action}, op_action:{op_action}')
        dist = np.linalg.norm(_state - env.center)
        pertub_dist = np.linalg.norm(state.detach().numpy() - env.center)

        if pertub_dist <= dist:
            #
            if loss <= effect:
                effect = loss
                # print(effect)
                attack = (state.detach().numpy() - _state)[0]
    if attack is None:
        return np.zeros_like(_state)
    else:
        # print(attack)
        return attack


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
    print(f'grey_c attack violation:{number_violate}, reach:{num_reached}')
    return grey_c_dist_list

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
    print(f'white attack violation:{number_violate}, reach:{num_reached}')
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
    print(f'black attack violation:{number_violate}, reach:{num_reached}')
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
    print(f'grey_s attack violation:{number_violate}, reach:{num_reached}')
    return grey_s_dist_list