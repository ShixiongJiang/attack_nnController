import math
import random
from typing import Optional

import gym
import numpy as np
from gym import spaces
from control.matlab import ss, lsim, linspace, c2d

from scipy.integrate import odeint
class DCEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        super(DCEnv, self).__init__()
        self.max_speed = -10
        self.max_torque = 10
        self.dt = 0.05

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.center = np.array([math.pi / 3.0 * 2, 0, 0])
        self.obstacle = np.array([[math.pi / 4.0, 0, 0], [math.pi / 8.0, 0, 0]])

        count = 0
        J = 0.01
        b = 0.1
        K = 0.01
        R = 1
        L = 0.5
        self.A = np.array([[0, 1, 0],
                           [0, -b / J, K / J],
                           [0, -K / L, -R / L]])
        self.A_dim = len(self.A)
        self.B = np.array([[0], [0], [1 / L]])
        self.xmeasure = 0
        self.attacked_element_idx = 0
        self.C = np.array([[1, 0, 0]])
        self.D = np.array([[0]])
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.2
        self.x_ref = np.array([[0], [0], [0]])  # 4th dim is remaining steps
        # store current trace
        self.cache1 = []
        self.cache2 = []
        self.cache3 = []
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[10]])
        # ball radius
        self.target_norm_radius = 0.6  # norm ball radius of target, tune this
        self.safe_norm_radius = 0.5  # norm ball radius of safe, tune this
        self.total_time = 120
        # step number
        self.steps = 0
        self.u_lowbound = None
        # store training traces
        self.state_array_1 = []
        self.state_array_2 = []
        self.state_array_3 = []
        self.caches = []
        self.reward_cache = []  # cache distances to target norm ball center
        self.avoid_reward_cache = []  # cache distances to obstacles norm ball center
        self.final_reward_cache = []  # cache final reward
        # How long should this trace be, i.e. deadline
        self.step_const = 50
        # Maximum reward from each trace
        self.max_reward_list = []
        self.quality_list = []
        self.total_steps = 0
        self.step_history = []
        self.sysc = ss(self.A, self.B, self.C, self.D)
        self.sysd = c2d(self.sysc, self.dt)

        high = np.array(self.x_ref.flatten(), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.reached = False
        self.k = 20

    def step(self, u):

        # simulate next step and get measurement
        self.steps += 1
        self.total_steps += 1
        terminated = False
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = 1
        yout, T, xout = lsim(self.sysc, self.last_u, [0, self.dt], self.xmeasure)
        self.xmeasure = xout[-1]

        # calculate euclidean distance and update reward cache
        dist = np.linalg.norm(self.xmeasure - self.center)
        state = self.xmeasure
        obs_dist = min(np.linalg.norm(state[:3] - self.obstacle[0]), np.linalg.norm(state[:3] - self.obstacle[1]))
        reward = self.target_norm_radius - dist
        obs_reward = obs_dist - self.safe_norm_radius

        self.reward_cache.append(reward)
        self.avoid_reward_cache.append(obs_reward)
        # quantitative semantics
        # reach reward, encourage reaching target
        if self.steps < 10:
            reach_reward = max(self.reward_cache)
        else:
            reach_reward = max(self.reward_cache[-10:])
        if self.steps < 10:
            avoid_reward = min(self.avoid_reward_cache)
        else:
            avoid_reward = min(self.avoid_reward_cache[-10:])

        #         # very strict reward, always within target
        #         strict_avoid_reward = avoid_reward - 0.5 * self.safe_norm_radius # half safe norm radius
        #         strict_reach_reward = reach_reward - 0.5 * self.target_norm_radius # half target norm radius

        # overall reward, pick one of the final_reward
        #         final_reward = reach_reward
        #         final_reward = approach_reward
        final_reward = min(reach_reward, avoid_reward)  # reach and avoid
        #         final_reward = min(approach_reward, avoid_reward) # approach and avoid
        #         final_reward = min(reach_reward, approach_reward) # reach and approach
        #         deadline_reward = (self.last_dist-dist)/(self.step_const - self.steps+1)
        #         final_reward = reach_reward
        # split cases: if already inside target, give very large constant reward for maintaining
        if dist <= self.target_norm_radius:
            final_reward = 10  # this gives 39/50 sucess with reach+approach+avoid

        self.final_reward_cache.append(final_reward)

        # update cached memory
        self.state = self.xmeasure
        self.state_array_1.append(self.state[0])
        self.state_array_2.append(self.state[1])
        self.state_array_3.append(self.state[2])
        self.cache1.append(self.state[0])
        self.cache2.append(self.state[1])
        self.cache3.append(self.state[2])
        self.last_dist = dist
        # If this is the last step, reset the state
        if self.steps == self.step_const or obs_dist <= self.safe_norm_radius:
            self.max_reward_list.append(max(self.final_reward_cache))  # use max final reward to measure episodes
            self.step_history.append(self.total_steps)
            self.quality_list.append(sum(self.final_reward_cache))
            terminated = True
            # self.reset()

        #         # If within target norm ball, early terminate
        #         if dist <= self.target_norm_radius:
        #             terminated = True
        #             self.reset()

        # Return next state, reward, done, info
        return self._get_obs(), final_reward, terminated, {}

    def reset(self):
        self.state = np.array([random.random() * math.pi - 1, random.random() * 2 - 1, random.random() * 20 - 10])
        self.reward_cache = []
        self.final_reward_cache = []
        self.steps = 0
        self.caches.append(self.cache1)
        self.caches.append(self.cache2)
        self.caches.append(self.cache3)
        self.cache1 = []
        self.cache2 = []
        self.cache3 = []
        self.xmeasure = self.state
        # random # of steps for this trace
        self.step_const = self.k  # deadline range
        self.reached = False
        return np.array(self.state)  # return something matching shape

    def _get_obs(self):
        current_state = list(self.state)
        return np.array(current_state)

    def render(self):
        return

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class adv_DCEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        super(adv_DCEnv, self).__init__()
        self.max_speed = -10
        self.max_torque = 10
        self.dt = 0.05

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.center = np.array([math.pi / 3.0 * 2, 0, 0])
        self.obstacle = np.array([[math.pi / 4.0, 0, 0], [math.pi / 8.0, 0, 0]])
        count = 0
        J = 0.01
        b = 0.1
        K = 0.01
        R = 1
        L = 0.5
        self.A = np.array([[0, 1, 0],
                           [0, -b / J, K / J],
                           [0, -K / L, -R / L]])
        self.A_dim = len(self.A)
        self.B = np.array([[0], [0], [1 / L]])
        self.xmeasure = 0
        self.attacked_element_idx = 0
        self.C = np.array([[1, 0, 0]])
        self.D = np.array([[0]])
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.2
        self.x_ref = np.array([[0], [0], [0]])  # 4th dim is remaining steps
        # store current trace
        self.cache1 = []
        self.cache2 = []
        self.cache3 = []
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[10]])
        # ball radius
        self.target_norm_radius = 0.5  # norm ball radius of target, tune this
        self.safe_norm_radius = 0.2  # norm ball radius of safe, tune this
        self.total_time = 120
        # step number
        self.steps = 0
        self.u_lowbound = None
        # store training traces
        self.state_array_1 = []
        self.state_array_2 = []
        self.state_array_3 = []
        self.caches = []
        self.reward_cache = []  # cache distances to target norm ball center
        self.avoid_reward_cache = []  # cache distances to obstacles norm ball center
        self.final_reward_cache = []  # cache final reward
        # How long should this trace be, i.e. deadline
        self.step_const = random.randint(10, 50)
        # Maximum reward from each trace
        self.max_reward_list = []
        self.quality_list = []
        self.total_steps = 0
        self.step_history = []
        self.sysc = ss(self.A, self.B, self.C, self.D)
        self.sysd = c2d(self.sysc, self.dt)

        high = np.array(self.x_ref.flatten(), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.reached = False
        self.k = 20

    def step(self, u):

        # simulate next step and get measurement
        self.steps += 1
        self.total_steps += 1
        terminated = False
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = 1
        yout, T, xout = lsim(self.sysc, self.last_u, [0, self.dt], self.xmeasure)
        self.xmeasure = xout[-1]

        # calculate euclidean distance and update reward cache
        dist = np.linalg.norm(self.xmeasure - self.center)
        state = self.xmeasure
        obs_dist = min(np.linalg.norm(state[:3] - self.obstacle[0]), np.linalg.norm(state[:3] - self.obstacle[1]))
        #         reward = self.target_norm_radius - dist
        obs_reward = obs_dist - self.safe_norm_radius
        obs_reward = -obs_reward
        #         self.reward_cache.append(reward)
        self.avoid_reward_cache.append(obs_reward)

        if self.steps < 10:
            avoid_reward = max(self.avoid_reward_cache)
        else:
            avoid_reward = max(self.avoid_reward_cache[-10:])
        final_reward = avoid_reward

        if obs_dist <= self.safe_norm_radius:
            final_reward = 10
        self.final_reward_cache.append(final_reward)

        # update cached memory
        self.state = self.xmeasure
        self.state_array_1.append(self.state[0])
        self.state_array_2.append(self.state[1])
        self.state_array_3.append(self.state[2])
        self.cache1.append(self.state[0])
        self.cache2.append(self.state[1])
        self.cache3.append(self.state[2])
        self.last_dist = dist
        # If this is the last step, reset the state
        if self.steps == self.step_const or obs_dist <= self.safe_norm_radius:
            self.max_reward_list.append(max(self.final_reward_cache))  # use max final reward to measure episodes
            self.step_history.append(self.total_steps)
            self.quality_list.append(sum(self.final_reward_cache))
            terminated = True
            self.reset()
        return self._get_obs(), final_reward, terminated, {}

    def reset(self):
        self.state = np.array([random.random() * math.pi, random.random() * 2 - 1, random.random() * 20 - 10])
        self.reward_cache = []
        self.final_reward_cache = []
        self.steps = 0
        self.caches.append(self.cache1)
        self.caches.append(self.cache2)
        self.caches.append(self.cache3)
        self.cache1 = []
        self.cache2 = []
        self.cache3 = []
        self.xmeasure = self.state
        # random # of steps for this trace
        self.step_const = self.k  # deadline range
        self.reached = False
        return np.array(self.state)  # return something matching shape

    def _get_obs(self):
        current_state = list(self.state)
        return np.array(current_state)

    def render(self):
        return

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class laa_DCEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        super(laa_DCEnv, self).__init__()
        self.max_speed = -10
        self.max_torque = 10
        self.dt = 0.05

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.center = np.array([math.pi / 3.0 * 2, 0, 0])
        self.obstacle = np.array([[math.pi / 4.0, 0, 0], [math.pi / 8.0, 0, 0]])

        count = 0
        J = 0.01
        b = 0.1
        K = 0.01
        R = 1
        L = 0.5
        self.A = np.array([[0, 1, 0],
                           [0, -b / J, K / J],
                           [0, -K / L, -R / L]])
        self.A_dim = len(self.A)
        self.B = np.array([[0], [0], [1 / L]])
        self.xmeasure = 0
        self.attacked_element_idx = 0
        self.C = np.array([[1, 0, 0]])
        self.D = np.array([[0]])
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.2
        self.x_ref = np.array([[0], [0], [0]])  # 4th dim is remaining steps
        # store current trace
        self.cache1 = []
        self.cache2 = []
        self.cache3 = []
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[10]])
        # ball radius
        self.target_norm_radius = 0.6  # norm ball radius of target, tune this
        self.safe_norm_radius = 0.5  # norm ball radius of safe, tune this
        self.total_time = 120
        # step number
        self.steps = 0
        self.u_lowbound = None
        # store training traces
        self.state_array_1 = []
        self.state_array_2 = []
        self.state_array_3 = []
        self.caches = []
        self.reward_cache = []  # cache distances to target norm ball center
        self.avoid_reward_cache = []  # cache distances to obstacles norm ball center
        self.final_reward_cache = []  # cache final reward
        # How long should this trace be, i.e. deadline
        self.step_const = 50
        # Maximum reward from each trace
        self.max_reward_list = []
        self.quality_list = []
        self.total_steps = 0
        self.step_history = []
        self.sysc = ss(self.A, self.B, self.C, self.D)
        self.sysd = c2d(self.sysc, self.dt)

        high = np.array(self.x_ref.flatten(), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.reached = False
        self.k = 20

    def step(self, u):

        # simulate next step and get measurement
        self.steps += 1
        self.total_steps += 1
        terminated = False
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = 1
        yout, T, xout = lsim(self.sysc, self.last_u, [0, self.dt], self.xmeasure)
        self.xmeasure = xout[-1]

        # calculate euclidean distance and update reward cache
        dist = np.linalg.norm(self.xmeasure - self.center)
        state = self.xmeasure
        obs_dist = min(np.linalg.norm(state[:3] - env.obstacle[0]), np.linalg.norm(state[:3] - env.obstacle[1]))
        reward = self.target_norm_radius - dist
        obs_reward = obs_dist - self.safe_norm_radius

        self.reward_cache.append(reward)
        self.avoid_reward_cache.append(obs_reward)
        # quantitative semantics
        # reach reward, encourage reaching target
        if self.steps < 10:
            reach_reward = max(self.reward_cache)
        else:
            reach_reward = max(self.reward_cache[-10:])
        if self.steps < 10:
            avoid_reward = min(self.avoid_reward_cache)
        else:
            avoid_reward = min(self.avoid_reward_cache[-10:])
        final_reward = -reward

        self.final_reward_cache.append(final_reward)

        # update cached memory
        self.state = self.xmeasure
        self.state_array_1.append(self.state[0])
        self.state_array_2.append(self.state[1])
        self.state_array_3.append(self.state[2])
        self.cache1.append(self.state[0])
        self.cache2.append(self.state[1])
        self.cache3.append(self.state[2])
        self.last_dist = dist
        # If this is the last step, reset the state
        if self.steps == self.step_const or obs_dist <= self.safe_norm_radius:
            self.max_reward_list.append(max(self.final_reward_cache))  # use max final reward to measure episodes
            self.step_history.append(self.total_steps)
            self.quality_list.append(sum(self.final_reward_cache))
            terminated = True

        return self._get_obs(), final_reward, terminated, {}

    def reset(self):
        self.state = np.array([random.random() * math.pi - 1, random.random() * 2 - 1, random.random() * 20 - 10])
        self.reward_cache = []
        self.final_reward_cache = []
        self.steps = 0
        self.caches.append(self.cache1)
        self.caches.append(self.cache2)
        self.caches.append(self.cache3)
        self.cache1 = []
        self.cache2 = []
        self.cache3 = []
        self.xmeasure = self.state
        # random # of steps for this trace
        self.step_const = self.k  # deadline range
        self.reached = False
        return np.array(self.state)  # return something matching shape

    def _get_obs(self):
        current_state = list(self.state)
        return np.array(current_state)

    def render(self):
        return

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False