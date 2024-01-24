import matplotlib.pyplot as plt
white_ave = []
black_ave = []
grey_c_ave = []
grey_s_ave = []

with open(r'white_dist.txt', 'r') as fp:
    white_ave = [float(x) -0.2 for x in fp.read().split()]
    # print(white_ave)

with open(r'black_dist.txt', 'r') as fp:
    black_ave = [float(x)-0.2 for x in fp.read().split()]

with open(r'grey_c_dist.txt', 'r') as fp:
    grey_c_ave = [float(x)-0.2 for x in fp.read().split()]

with open(r'grey_s_dist.txt', 'r') as fp:
    grey_s_ave = [float(x) -0.2for x in fp.read().split()]

# fig = plt.figure()
#
# fig.set_figheight(3)
# fig.set_figwidth(9)


import seaborn as sns
fig, ax = plt.subplots()
fig.set_figheight(4.6)
fig.set_figwidth(12)
plt.yticks(fontsize=44)
plt.xticks(fontsize=44)
# plt.show()
# plt.ylabel('Robustness of safety', fontsize=16)
data = [white_ave, grey_c_ave, grey_s_ave, black_ave]
B = ax.boxplot(data, medianprops=dict(linewidth=3), labels=("WB","GB-C","GB-P", 'BB'))
middle = [item.get_ydata()[1] for item in B['medians']]
plt.ylabel('Avg Dist', fontsize=44)

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
# plt.savefig('stealthy.png', dpi=500)
plt.savefig('box_pointgoal.pdf', bbox_inches='tight', dpi=500)

plt.show()


