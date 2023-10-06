from stable_baselines3 import A2C, SAC, PPO
import torch as th
from customerEnv import SafetyCarCircle
env = SafetyCarCircle(render_mode='human')
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64, 64], vf=[64, 64]))

obs, info = env.reset()
model = PPO.load('SafetyCarCirclePPO-1.zip', env=env)

for _ in range(1000):
    # action = env.action_space.sample()  # this is where you would insert your policy
    action, _state = model.predict(obs, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        observation, info = env.reset()
env.close()


