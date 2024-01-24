import matplotlib.pyplot as plt
import numpy as np
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

# fig = plt.figure()
#
# fig.set_figheight(3)
# fig.set_figwidth(9)


fig, ax = plt.subplots()
fig.set_figheight(4.6)
fig.set_figwidth(12)
plt.yticks(fontsize=44)
plt.xticks(fontsize=44)
# ax.yaxis.set_ticks(np.arange(-0.1, 2.4, 0.4))
data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
B = ax.boxplot(data, medianprops=dict(linewidth=3), labels=("WB","GB-C","GB-P", 'BB'))
plt.ylabel('Avg Dist', fontsize=44)
middle = [item.get_ydata()[1] for item in B['medians']]
ax.yaxis.set_ticks(np.arange(0.5, 4, 1))
plt.ylim(0.5, 4)

# plt.legend()
plt.savefig('box_Bicycle.pdf', bbox_inches='tight', dpi=500)

plt.show()


