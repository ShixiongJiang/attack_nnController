import safety_gymnasium

if __name__ == '__main__':
   # env = safety_gymnasium.vector.make("SafetyCarCircle1-v0", render_mode="human")
   env_id = 'SafetyPointGoal1-v0'
   safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode='human')
   env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
   observation, info = env.reset(seed=0)
   print(env.unwrapped._agent().pos)
   # print(env.unwrapped.unwrapped)
   for _ in range(10000):
      action = env.action_space.sample()  # this is where you would insert your policy
      observation, reward, terminated, truncated, info = env.step(action)

      if terminated or truncated:
         observation, info = env.reset()

         print(env.unwrapped._agent().pos)
   env.close()