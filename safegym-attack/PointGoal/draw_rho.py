


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statistics
# print(len(rho_ave))
# print((rho_list[0]))
# print(rho_list[1])

with open(r'rho.txt', 'r') as fp:
    white_rho_ave = [float(x)  for x in fp.read().split()]
    # print(white_ave)

with open(r'pertub_rho.txt', 'r') as fp:
    white_pertub_rho_ave = [float(x) for x in fp.read().split()]

with open(r'grey_c_rho.txt', 'r') as fp:
    grey_c_ave = [float(x)  for x in fp.read().split()]
    # print(white_ave)
with open(r'grey_c_pertub_rho.txt', 'r') as fp:
    grey_c_pertub_rho_ave = [float(x)for x in fp.read().split()]

with open(r'grey_s_rho.txt', 'r') as fp:
    grey_s_rho_ave = [float(x) for x in fp.read().split()]
    # print(white_ave)
with open(r'grey_s_pertub_rho.txt', 'r') as fp:
    grey_s_pertub_rho_ave = [float(x) for x in fp.read().split()]
with open(r'black_rho.txt', 'r') as fp:
    black_rho_ave = [float(x)for x in fp.read().split()]
    # print(white_ave)
with open(r'black_pertub_rho.txt', 'r') as fp:
    black_pertub_rho_ave = [float(x)for x in fp.read().split()]
fig, ax = plt.subplots()
fig.set_figheight(3.8)
fig.set_figwidth(9.5)
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
plt.xlabel('Time[sec]')

# plt.boxplot([rho_ave, pertub_rho_ave], showfliers=False)
# plt.plot( np.arange(0, len(rho_ave))*0.05 , np.array(pertub_rho_ave) - np.array(rho_ave), linewidth=3,color='gold')
from scipy.ndimage.filters import gaussian_filter1d
# ax.yaxis.set_ticks(np.arange(-1.5, 0.2, 0.4))
rho_ave = [white_rho_ave, grey_c_ave, grey_s_rho_ave, black_rho_ave]
pertub_rho_ave = [white_pertub_rho_ave, grey_c_pertub_rho_ave, grey_s_pertub_rho_ave, black_pertub_rho_ave]
ysmoothed = []
for i in pertub_rho_ave:

    ysmoothed.append(gaussian_filter1d(i, sigma=2))
smoothed = []
for i in rho_ave:
    smoothed.append(gaussian_filter1d(i, sigma=2))
rho_ave = smoothed

# color = ['purple', 'grey', 'lawngreen', 'black']
label = ['WB','GB-C', 'GB-P', 'BB']
# plt.tick_params(right = False , labelleft = '' )

for i in range(0,4):
    m = i*0.5 - 1.5
    p = plt.plot( np.arange(0, len(pertub_rho_ave[i][0:200]))*0.05 , ysmoothed[i][0:200] +0.001 + m, linewidth=2,
               linestyle='dotted')

    plt.plot( np.arange(0, len(pertub_rho_ave[i][0:200]))*0.05, rho_ave[i][0:200]+ m, linewidth=2, color = p[0].get_color(), label=label[i])
# listOf_Yticks = np.arange(-16, 2, 4) * 0.1
# plt.yticks(listOf_Yticks)# plt.xlabel('The average robustness of ')
plt.ylabel('Robustness of $\phi_g$', fontsize= 22)

plt.xlabel('Time[sec]', fontsize = 22)
ax.xaxis.set_label_coords(0.48, -0.15)
# plt.legend( loc='upper left', prop={'size': 22})
plt.legend(loc='lower right', prop={'size': 22})
plt.savefig('rho_pointgoal.pdf', bbox_inches='tight', dpi=500)
plt.ylim(-2.9, -0.1)
plt.show()
