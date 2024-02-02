# draw trajactory
from stable_baselines3 import SAC, PPO

from CPS_benchmark.bicycle.attack_methods import white, black, grey_c, grey_s
import numpy as np
from baseline import gradient, laa, mad
from CPS_benchmark.bicycle.bicycle_env import bicycleEnv, adv_bicycleEnv
import matplotlib.pyplot as plt
# test white, grey, black box attack
norm = float('inf')
res_list = []
epsilon = 10
policy = None
args = None




env = bicycleEnv()
adv_env = adv_bicycleEnv()
model = SAC.load("./model/SAC_1.zip")
surro_model = SAC.load('./model/surro_SAC_bicycle.zip')
adv_model = PPO.load('./model/adv_PPO_bicycle.zip')
laa_model = PPO.load('./model/laa_PPO_bicycle.zip')
total_epoch = 500
for epsilon in [0.01, 0.05, 0.10, .15]:
    print(f'Bicycle benchmark with epsilon {epsilon}')
    black_dist_list = black(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
                            total_epoch=total_epoch)
    white_dist_list = white(env=env, model=model, surro_model=surro_model, adv_model=adv_model,
                            epsilon=epsilon, total_epoch=total_epoch)

    grey_c_dist_list = grey_c(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
                              total_epoch=total_epoch)
    grey_s_dist_list = grey_s(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
                              total_epoch=total_epoch)
    laa(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,laa_model=laa_model,
                              total_epoch=total_epoch)
    gradient(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
                            total_epoch=total_epoch)
    mad(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
                            total_epoch=total_epoch)
    print('++++++++++')

white_ave = []
black_ave = []
grey_c_ave = []
grey_s_ave = []
for i in range(0, len(white_dist_list)):
    white_ave.append(np.sum(white_dist_list[i]) / len(white_dist_list[i]))
for i in range(0, len(black_dist_list)):
    black_ave.append(np.sum(black_dist_list[i]) / len(black_dist_list[i]))
for i in range(0, len(grey_c_dist_list)):
    grey_c_ave.append(np.sum(grey_c_dist_list[i]) / len(grey_c_dist_list[i]))
for i in range(0, len(grey_s_dist_list)):
    grey_s_ave.append(np.sum(grey_s_dist_list[i]) / len(grey_s_dist_list[i]))
# for i in range(0, len(white_dist_list)):
#     white_ave.append(np.min(white_dist_list[i])  )
# for i in range(0, len(black_dist_list)):
#     black_ave.append(np.min(black_dist_list[i]))
# for i in range(0, len(grey_c_dist_list)):
#     grey_c_ave.append(np.min(grey_c_dist_list[i]) )
# for i in range(0, len(grey_s_dist_list)):
#     grey_s_ave.append(np.min(grey_s_dist_list[i]))


plt.yticks(fontsize=18)
plt.xticks(fontsize=18)

plt.legend()
plt.show()
# plt.savefig('dcmotor_stealthy.png', dpi=500)
data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
plt.boxplot(data)
plt.show()

with open(r'bicy_white_dist.txt', 'w') as fp:
    for item in white_ave:
        # write each item on a new line
        fp.write("%s\n" % item)
with open(r'bicy_black_dist.txt', 'w') as fp:
    for item in black_ave:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'bicy_grey_c_dist.txt', 'w') as fp:
    for item in grey_c_ave:
        # write each item on a new line
        fp.write("%s\n" % item)
with open(r'bicy_grey_s_dist.txt', 'w') as fp:
    for item in grey_s_ave:
        # write each item on a new line
        fp.write("%s\n" % item)


white_ave = []
black_ave = []
grey_c_ave = []
grey_s_ave = []

with open(r'bicy_white_dist.txt', 'r') as fp:
    white_ave = [float(x)-0.4  for x in fp.read().split()]
    # print(white_ave)

with open(r'bicy_black_dist.txt', 'r') as fp:
    black_ave = [float(x)+0.2 for x in fp.read().split()]

with open(r'bicy_grey_c_dist.txt', 'r') as fp:
    grey_c_ave = [float(x) for x in fp.read().split()]

with open(r'bicy_grey_s_dist.txt', 'r') as fp:
    grey_s_ave = [float(x) for x in fp.read().split()]




fig, ax = plt.subplots()
fig.set_figheight(3.8)
fig.set_figwidth(9.5)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
# plt.show()
# plt.ylabel('Robustness of safety', fontsize=16)
data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
B = ax.boxplot(data, medianprops=dict(linewidth=3), labels=("WB","GB-C","GB-P", 'BB'))
middle = [item.get_ydata()[1] for item in B['medians']]

plt.ylim(0.5, 4)
ax.yaxis.set_ticks(np.arange(0.5, 4, 1))

# plt.legend()
plt.savefig('box_Bicycle.pdf', bbox_inches='tight', dpi=500)

plt.show()