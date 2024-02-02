from pointgoal_env import PointGoal1
from stable_baselines3 import PPO, SAC
from attack_methods import black, white, grey_c, grey_s
from utils.baseline_attack import gradient, mad, laa
env = PointGoal1()
model = PPO.load('./model/PPO_PointGoal.zip')
surro_model = SAC.load('./model/surro_SAC_PointGoal.zip')
adv_model = PPO.load('./model/adv_PPO_PointGoal.zip')
laa_model = PPO.load('./model/laa_PPO_PointGoal.zip')

obs, info = env.reset()
total_epoch = 500
env = PointGoal1()

for epsilon in [0.01, 0.05, 0.15, 0.1]:
# for epsilon in [0.05]:
    print(f'PointGoal Benchmark with epsilon {epsilon}')
    black_dist_list = black(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    white_dist_list = white(env=env, model=model, surro_model=surro_model, adv_model=adv_model,
                                                 epsilon=epsilon, total_epoch=total_epoch)
    grey_c_dist_list = grey_c(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon , total_epoch= total_epoch)
    grey_s_dist_list = grey_s(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    gradient(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    mad(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, total_epoch= total_epoch)
    laa(env=env,  model=model, surro_model=surro_model, adv_model=adv_model, laa_model=laa_model, epsilon=epsilon, total_epoch= total_epoch)
    print('++++++++++++')

import matplotlib.pyplot as plt
import numpy as np
white_ave = []
black_ave = []
grey_c_ave = []
grey_s_ave = []

for i in range(0, len(white_dist_list)):
    white_ave.append(np.min(white_dist_list[i]))
for i in range(0, len(black_dist_list)):
    black_ave.append(np.min(black_dist_list[i]))
for i in range(0, len(grey_c_dist_list)):
    grey_c_ave.append(np.min(grey_c_dist_list[i]))
for i in range(0, len(grey_s_dist_list)):
    grey_s_ave.append(np.min(grey_s_dist_list[i]))




fig, ax = plt.subplots()
fig.set_figheight(4.6)
fig.set_figwidth(12)
plt.yticks(fontsize=44)
plt.xticks(fontsize=44)

data = [white_ave[0], grey_c_ave[0], grey_s_ave[0], black_ave[0]]
print(white_ave)
B = ax.boxplot(data, medianprops=dict(linewidth=3), labels=("WB","GB-C","GB-P", 'BB'))
middle = [item.get_ydata()[1] for item in B['medians']]
plt.ylabel('Avg Dist', fontsize=44)

plt.savefig('./data/box_pointgoal.pdf', bbox_inches='tight', dpi=500)

plt.show()


