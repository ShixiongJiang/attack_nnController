from customEnv.env import SafetyPointGoal1
from stable_baselines3 import PPO
# from baseline import FGSM
env = SafetyPointGoal1(render_mode='human')
model = PPO.load('train/model/SafetyPointGoal1-PPO-7.zip', env=env)
# adv_model = PPO.load('train/SafetyPointGoal1-PPO-5.zip', env=env)
epoch = 0
reach = 0
violate = 0
for i in range(5):
    epoch = 0
    reach = 0
    violate = 0
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        # attack = white_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        # print(attack)
        # obs = attack + obs

            # state = FGSM(env, obs, model, surro_model=model, adv_model=None, epsilon=0.5)
            # print(state)
            # print(obs)
            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
            # obs = state
            action, _state = model.predict(obs, deterministic=True)
            # action[0] = action[0] / 20


            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:

                epoch +=1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1
                # obs, info = env.reset()
                break

        if epoch >= 100:
            break
    print(f'clean attack violation:{violate}, reach:{ reach}')