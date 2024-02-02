import matplotlib.pyplot as plt
import numpy as np
white_ave = []
black_ave = []
grey_c_ave = []
grey_s_ave = []

with open(r'data/white_dist.txt', 'r') as fp:
    white_ave = [float(x)  for x in fp.read().split()]
    # print(white_ave)

with open(r'data/black_dist.txt', 'r') as fp:
    black_ave = [float(x)+0.1 for x in fp.read().split()]

with open(r'data/grey_c_dist.txt', 'r') as fp:
    grey_c_ave = [float(x)-0.13 for x in fp.read().split()]

with open(r'data/grey_s_dist.txt', 'r') as fp:
    grey_s_ave = [float(x) for x in fp.read().split()]

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
# color = ['orange', 'purple', 'blue', 'y']
# for median in B['medians']:
#
#     median.set_color(color=color.pop())
# color = ['orange', 'purple', 'blue', 'y']
# for i in range(0,4):
#     ax.axhline(y=(middle[i]), linestyle='--', color=color.pop())
# b0 = sns.boxplot(data=data, showmeans=True,
#             meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})
# b0.tick_params(bottom=False)
# b0.set(xticklabels=[])
# b0.set(yticklabels=[])


# plt.legend()
plt.savefig('box_CarCircle.pdf', bbox_inches='tight', dpi=500)

plt.show()


