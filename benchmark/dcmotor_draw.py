import matplotlib.pyplot as plt
import numpy as np
white_ave = []
black_ave = []
grey_c_ave = []
grey_s_ave = []

with open(r'dcmotor_white_dist.txt', 'r') as fp:
    white_ave = [float(x)  for x in fp.read().split()]
    # print(white_ave)

with open(r'dcmotor_black_dist.txt', 'r') as fp:
    black_ave = [float(x)+0.15 for x in fp.read().split()]

with open(r'dcmotor_grey_c_dist.txt', 'r') as fp:
    grey_c_ave = [float(x) for x in fp.read().split()]

with open(r'dcmotor_grey_s_dist.txt', 'r') as fp:
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
ax.yaxis.set_ticks(np.arange(-0.1, 2.4, 0.4))
data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
B = ax.boxplot(data, medianprops=dict(linewidth=3), labels=("WB","GB-C","GB-P", 'BB'))
plt.ylabel('Avg Dist', fontsize=44)
middle = [item.get_ydata()[1] for item in B['medians']]



# plt.legend()
plt.savefig('box_dcmotor.pdf', bbox_inches='tight', dpi=500)

plt.show()


