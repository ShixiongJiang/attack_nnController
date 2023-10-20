import math

import numpy as np
import torch
from scipy.stats import beta
def Gradient_attack(env, state, model, surro_model, adv_model, epsilon, dim):
    # print(type(state))
    _action = model.predict(state)[0]
    _state = state
    action = model.predict(state)[0]
    state_range = _state / np.linalg.norm(_state) * epsilon * math.sqrt(44)

    for i in range(10):
        state = torch.tensor(state)
        # print(state)
        state.requires_grad_(True)
        Q = model.policy.forward(state.view(1, -1))[1]

        # q_vals = (model.critic(state, action))
        Q.backward()
        grad = state.grad
        grad_dir = grad / torch.norm(grad) * epsilon * dim

        state = state - grad_dir

        state_range = grad / np.linalg.norm(grad) * epsilon * math.sqrt(44)
        state = torch.clip(state, torch.from_numpy(_state) - state_range, torch.from_numpy(_state) + state_range)

        Q_adv = model.policy.forward(state.view(1, -1))[1]
    if Q_adv < Q:
        # print("attack")
        return state.detach().numpy()
    else:
        # print("no attack")
        return _state

    # print(grad)
    # sign_data_grad = data_grad.sign()
    # perturbed_image = image + epsilon * sign_data_grad
    # # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # # Return the perturbed image
    # return perturbed_image