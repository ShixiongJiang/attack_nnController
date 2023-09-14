import safety_gymnasium
env = safety_gymnasium.vector.make("SafetyCarCircle1-v0", render_mode="human")
observation, info = env.reset(seed=0)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()