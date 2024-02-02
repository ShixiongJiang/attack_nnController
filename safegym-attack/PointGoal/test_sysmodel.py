import numpy as np
from control.matlab import ss, lsim, linspace, c2d
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from customEnv.env import SafetyPointGoal1


def get_op_action(obs):
    obstacle = np.argmax(obs[28:44])
    hazard_dist = 3 - 3 * obs[28:44].max()
    turn = obs[4]
    velo = obs[3]
    if 4 < obstacle < 12:
        action0 = -0.05 * hazard_dist
    else:
        action0 = 0.05 * hazard_dist
    if 4 <= obstacle <= 12:
        if obstacle < 8:
            action1 = -1
        elif obstacle > 8:
            action1 = 1
        else:
            action1 = 0
    else:
        if 1 <=obstacle <= 3:
            action1 = 1
        elif 13 <= obstacle <= 15:
            action1 = -1
        else:
            action1 = 0
    action = [action0, action1]
    return action

class sys_pointgoal:
    def __init__(self):
        self.state = None
        self.xmeasure = None
        self.last_u = None
        sys_model = np.load('./data/estimated_model_pointgoal.npz')
        self.A = sys_model['A']
        self.B = sys_model['B']
        self.C = np.ones(shape=(1, 44))
        self.D = np.zeros(shape=(1, 2))
        self.sysc = ss(self.A, self.B, self.C, self.D)
        self.dt = 0.002

        self.sysd = c2d(self.sysc, self.dt)


    def step(self, u, state):
        self.xmeasure = state
        self.last_u = u
        yout, T, xout = lsim(self.sysc, self.last_u, [0, self.dt], self.xmeasure)
        self.xmeasure = xout[-1]

    def calculate_mali_action(self, state):
        def fun(x):
            # for i in range(1):
            next_obs = self.sysd.A @ state+ self.sysd.B @ x
            # next_obs = self.sysc.A @ state+ self.sysc.B @ x
            hazard_dist = 3 - 3 * next_obs[0, 28:44].max()
            # hazard_dist = 3 - 3 * max(next_obs[0, 42:44].max(), next_obs[0, 0:2].max())
            # print(hazard_dist)
            return hazard_dist

        x_start = [-1, 0]
        lim = [-1, 1]

        bounds = Bounds([lim[0], lim[0]], [lim[1], lim[1]])
        # self.state = state
        # non_linear_eq = lambda x: 3 - 3 * max((self.sysd.A @ state+ self.sysd.B @ x)[0, 42:44].max(), (self.sysd.A @ state+ self.sysd.B @ x)[0, 0:2].max())
        non_linear_eq = lambda x: (self.sysd.A @ state + self.sysd.B @ x)[0, 28:44].max()
        non_linear_constr = NonlinearConstraint(non_linear_eq, -np.inf, 3)
        result = minimize(fun, x_start, bounds=bounds, constraints=non_linear_constr)
        op_action = (result.x)
        # print(result.fun)
        return op_action


env = SafetyPointGoal1(render_mode='human')
sys = sys_pointgoal()
obs, info = env.reset()
last_action = None
ac_diff = 0.0002
print(sys.A)
for i in range(10000):
    action = sys.calculate_mali_action(obs)
    op_action = get_op_action(obs)
    action[0] = op_action[0]
    # action = op_action
    print(action)
    obs, reward, done, trun, info = env.step(action)
    hazard_dist = 3 - 3 * max(obs[28:44])

    if done or trun:
        obs, info = env.reset()


