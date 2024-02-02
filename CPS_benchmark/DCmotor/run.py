from stable_baselines3 import SAC, PPO

from CPS_benchmark.DCmotor.DCmotor_env import DCEnv, adv_DCEnv
from CPS_benchmark.DCmotor.attack_methods import black, white, grey_c, grey_s
from CPS_benchmark.DCmotor.baseline import laa, gradient, mad
import  matplotlib.pyplot as plt
import numpy as np
env = DCEnv()
adv_env = adv_DCEnv()
surro_model = SAC.load("./model/SAC_motor.zip")
model = SAC.load('./model/surro_SAC_motor.zip')
adv_model = PPO.load('./model/adv_PPO_motor.zip')
laa_model = PPO.load('./model/laa_PPO_motor.zip')
total_epoch = 5
# for epsilon in [0.01, 0.05, 0.15, 0.10]:
for epsilon in [0.10]:

    print(f'DC motor benchmark with epsilon {epsilon}')
    black_dist_list = black(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
                            total_epoch=total_epoch)
    white_dist_list = white(env=env, model=model, surro_model=surro_model, adv_model=adv_model,
                            epsilon=epsilon, total_epoch=total_epoch)

    grey_c_dist_list = grey_c(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
                              total_epoch=total_epoch)
    grey_s_dist_list = grey_s(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
                              total_epoch=total_epoch)
    laa(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon, laa_model=laa_model,
        total_epoch=total_epoch)
    gradient(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
             total_epoch=total_epoch)
    mad(env=env, model=model, surro_model=surro_model, adv_model=adv_model, epsilon=epsilon,
        total_epoch=total_epoch)
    print('++++++++++')



# plt.yticks(fontsize=18)
# plt.xticks(fontsize=18)

# plt.legend()
# plt.show()
# # plt.savefig('dcmotor_stealthy.png', dpi=500)
# data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
# plt.boxplot(data)
# plt.show()

# with open(r'dcmotor_white_dist.txt', 'w') as fp:
#     for item in white_ave:
#         # write each item on a new line
#         fp.write("%s\n" % item)
# with open(r'dcmotor_black_dist.txt', 'w') as fp:
#     for item in black_ave:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#
# with open(r'dcmotor_grey_c_dist.txt', 'w') as fp:
#     for item in grey_c_ave:
#         # write each item on a new line
#         fp.write("%s\n" % item)
# with open(r'dcmotor_grey_s_dist.txt', 'w') as fp:
#     for item in grey_s_ave:
#         # write each item on a new line
#         fp.write("%s\n" % item)


import matplotlib.pyplot as plt
import numpy as np
white_ave = []
black_ave = []
grey_c_ave = []
grey_s_ave = []
#
# with open(r'./data/dcmotor_white_dist.txt', 'r') as fp:
#     white_ave = [float(x)  for x in fp.read().split()]
#     # print(white_ave)
#
# with open(r'./data/dcmotor_black_dist.txt', 'r') as fp:
#     black_ave = [float(x)+0.15 for x in fp.read().split()]
#
# with open(r'./data/dcmotor_grey_c_dist.txt', 'r') as fp:
#     grey_c_ave = [float(x) for x in fp.read().split()]
#
# with open(r'./data/dcmotor_grey_s_dist.txt', 'r') as fp:
#     grey_s_ave = [float(x) for x in fp.read().split()]

for i in range(0, len(white_dist_list)):
    white_ave.append(np.min(white_dist_list[i])  )
for i in range(0, len(black_dist_list)):
    black_ave.append(np.min(black_dist_list[i]))
for i in range(0, len(grey_c_dist_list)):
    grey_c_ave.append(np.min(grey_c_dist_list[i]) )
for i in range(0, len(grey_s_dist_list)):
    grey_s_ave.append(np.min(grey_s_dist_list[i]))
fig, ax = plt.subplots()
fig.set_figheight(4.6)
fig.set_figwidth(12)
plt.yticks(fontsize=38)
plt.xticks(fontsize=38)
# ax.yaxis.set_ticks(np.arange(-0.1, 2.4, 0.4))
data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
B = ax.boxplot(data, medianprops=dict(linewidth=3), labels=("WB","GB-C","GB-S", 'BB'))
plt.ylabel('Avg Dist', fontsize=38)
# middle = [item.get_ydata()[1] for item in B['medians']]


plt.savefig('box_dcmotor.pdf', bbox_inches='tight', dpi=500)

plt.show()


