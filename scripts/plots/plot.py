import numpy as np
import pandas as pd
import seaborn as sns
from scripts.common.utils import config_pyplot

sns.set_context("paper")
plt = config_pyplot(font_size=20, tick_size=20)
sns.set_style("ticks") # , {'axes.edgecolor': '#cccccc'})

def plot_violin(names, means, hist_data, x_label, y_label):
    """@:param hist_data: a list of lists, for each name a list of metric values. """
    performance_data = []
    for i_arch in range(len(names)):
        for ret in hist_data[i_arch]:
            performance_data.append((names[i_arch], means[i_arch], ret))

    performance_df = pd.DataFrame(performance_data, columns=[x_label, 'Mean Architecture Performance', y_label])
    # plot violins
    sns.violinplot(x=x_label, y=y_label, data=performance_df, inner='stick', bw=0.5)
    # plot means
    plt.scatter(names, means, linestyle='None', color='#ffffff', s=60)
    # plot baseline
    plt.plot(np.arange(-1, len(names) + 1), np.ones((len(names) + 2,)) * means[0], c='#777777',
             linestyle='--', linewidth=1)
    plt.xlim([-0.5, len(names)-0.5])
    ymin, ymax = np.min(hist_data), np.max(hist_data)
    margin = 0.1*(ymax - ymin)
    plt.ylim([ymin - margin, ymax + margin])
    # plt.yticks(np.linspace(275, 310, 8), np.linspace(275, 310, 8, dtype=int))
    plt.xlabel(x_label)


norm_deltas = np.array([5209568, 5209664, 5810848, 6612304])/1e6
bsln = np.array([16000000, 16000000, 16000000, 16000000])/1e6
norm_angs = np.array([9417888, 9517888, 9000000, 9900000])/1e6
aps = [bsln, norm_angs, norm_deltas]
means = [np.mean(ap) for ap in aps]

plot_violin(['Baseline', 'Normalized\nAngles', 'Normalized\nAngle Deltas'],
            means, aps, '', 'Training Timesteps [$x10^6$]')
plt.show()