from utils.baselines import laa_attack, MAD, Gradient_attack
def laa(env, model, surro_model, adv_model,laa_model, epsilon, total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    while True:
        obs, info = env.reset()
        while True:
            attack = laa_attack(env, obs, model=model, surro_model=surro_model, adv_model=adv_model, laa_model=laa_model, epsilon=epsilon)
            obs = attack + obs
            action, _state = model.predict(obs, deterministic=True)

            obs, reward, done, trun, info = env.step(action)
            # print(obs[0:12])
            if done or trun:

                epoch += 1
                goal_dist = 3 - 3 * max(obs[12:28])
                obs_dist = 3 - 3 * max(obs[28:44])
                if goal_dist < 0.4:
                    reach += 1
                elif obs_dist < 0.2:
                    violate += 1

                break
        if epoch >= total_epoch:
            break
    print(f'laa attack violation:{violate}, reach:{reach}')

def mad(env,  model, surro_model, adv_model, epsilon,  total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        while True:
            attack = MAD(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            # print(attack - obs)
            obs = attack + obs

            action, _state = model.predict(obs, deterministic=True)
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
        if epoch >= total_epoch:
            break
    print(f'mad attack violation:{violate}, reach:{ reach}')


def gradient(env,  model, surro_model, adv_model, epsilon,  total_epoch):
    epoch = 0
    reach = 0
    violate = 0
    while True:
        # attack = black_attack(env, obs, model, surro_model=model, adv_model=adv_model, epsilon=0.5)
        obs, info = env.reset()
        while True:
            attack = Gradient_attack(env, obs, model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon)
            # print(attack)
            obs = obs + attack

            action, _state = model.predict(obs, deterministic=True)


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
        if epoch >= total_epoch:
            break
    print(f'Gradient attack violation:{violate}, reach:{ reach}')