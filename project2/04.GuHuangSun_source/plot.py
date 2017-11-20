'''
Generates plots for the Spearman's and Kendall's Coefficients as well as their
p-values, using the results from Matlab
'''

import numpy as np
import matplotlib.pyplot as plt

# Plot for Google's PageRank
alpha = np.sort(np.append(np.arange(0.1, 1, 0.1), 0.85))
spearman = np.array([0.6644, 0.6671, 0.6751, 0.6803, 0.6857, 0.6972, 0.6938, \
    0.7033, 0.7030, 0.7021])
kendall = np.array([0.4792, 0.4828, 0.4877, 0.4934, 0.4983, 0.5061, 0.5033, \
    0.5104, 0.5104, 0.5083])
spearman_pval = np.array([0.6015, 0.4704, 0.2260, 0.1391, 0.0822, 0.0262, \
    0.0371, 0.0140, 0.0144, 0.0159]) * 1e-10
kendall_pval = np.array([0.1299, 0.0982, 0.0661, 0.0418, 0.0279, 0.0147, \
    0.0186, 0.0103, 0.0103, 0.0123]) * 1e-8

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16,12), dpi=100)
ax[0].plot(alpha, spearman, color='orange', label="Spearman", linewidth=3.)
ax[0].plot(alpha, kendall, color='blue', label="Kendall", linewidth=3.)
ax[0].set_title("Correlation between Google's Pagerank and Actual Ranking\n", \
    size=24, family='serif', weight='bold')
ax[1].plot(alpha, spearman_pval, color='orange', label="Spearman", linewidth=3.)
ax[1].plot(alpha, kendall_pval, color='blue', label="Kendall", linewidth=3.)
ax[0].tick_params(labelsize=20)
ax[1].tick_params(labelsize=20)
ax[0].set_ylabel('Coefficient', size=20)
ax[1].set_ylabel(r'$p$ Value', size=20)
plt.xticks(fontsize=16)
plt.xlabel(r'$\alpha$', size=20)
legend = ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.03), \
          fancybox=True, shadow=True, ncol=2)
for label in legend.get_texts():
    label.set_fontsize(16)
plt.show()
plt.clf()


# Plot for HITS ranking
loc = np.array([0, 1])
method = np.array(['Hub', 'Authority'])
spearman = np.array([0.5418, 0.7487])
kendall = np.array([0.3871, 0.5741])
spearman_pval = np.array([426.19, 0]) * 1e-10
kendall_pval = np.array([0.95525, 0]) * 1e-8
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12,16), dpi=100)
rects00 = ax[0].bar(loc-.1, spearman, width=.2, color='orange')
rects01 = ax[0].bar(loc+.1, kendall, width=.2, color='blue')
rects10 = ax[1].bar(loc-.1, spearman_pval, width=.2, color='orange')
rects11 = ax[1].bar(loc+.1, kendall_pval, width=.2, color='blue')
ax[0].set_ylabel('Coefficients', size=20)
ax[0].set_title("Correlation between HITS and Actual Ranking\n", \
    size=30, family='serif', weight='bold')
ax[0].set_xticks(loc)
ax[0].set_yticks(np.arange(0, 1, 0.1))
ax[1].set_ylabel(r'$p$ Value', size=20)
ax[1].set_xticklabels(method)
ax[1].set_yticks(np.arange(0, 6e-8, 1e-8))
ax[0].tick_params(labelsize=24)
ax[1].tick_params(labelsize=24)
legend = ax[1].legend((rects10[0], rects11[0]), ('Spearman', 'Kendall'), loc='lower center', \
    bbox_to_anchor=(0.5, 1.03), fancybox=True, shadow=True, ncol=2)
for label in legend.get_texts():
    label.set_fontsize(20)
def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, \
            '{}'.format(height), ha='center', va='bottom', size=18)

for i in [0, 1]:
    for j in [0, 1]:
        exec('autolabel(ax[{}], rects{}{})'.format(i, i, j))
plt.show()
