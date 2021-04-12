import numpy as np
import pandas as pd
import seaborn as sns
from scripts.common.utils import config_pyplot

sns.set_context("paper")
plt = config_pyplot(font_size=20, tick_size=20)
# sns.set_style("ticks") # , {'axes.edgecolor': '#cccccc'})


def plot_violin(names, means, hist_data, x_label='', y_label='', text_size=18):
    """@:param hist_data: a list of lists, for each name a list of metric values. """
    performance_data = []
    for i_arch in range(len(names)):
        for ret in hist_data[i_arch]:
            performance_data.append((names[i_arch], means[i_arch], ret))

    performance_df = pd.DataFrame(performance_data, columns=[x_label, 'Mean Architecture Performance', y_label])

    # plot violins
    violin = sns.violinplot(x=x_label, y=y_label, data=performance_df, inner='stick',
                            bw=0.5, width=1.25, linewidth=1.5)

    # plot horizontal lines indicating the means
    for i in range(len(means)):
        color = sns.color_palette()[i]
        # plt.plot(np.arange(-1, len(names) + 1), np.ones((len(names) + 2,)) * means[i], c=color,
        #          linestyle='--', linewidth=2, zorder=0)
        plt.plot(plt.gca().get_xlim(), np.ones(2) * means[i], c=color,
                 linestyle='--', linewidth=2, zorder=0)

        vertical_offset = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:-5.5e5}[i]
        plt.gca().text(-0.45, means[i]+1e5+vertical_offset,
                       f'{np.round(means[i]/1e6, 1)}M',
                       fontsize=text_size-1, color=color)

    # plot means
    plt.scatter(names, means, linestyle='None', color='#ffffff', s=40, zorder=10)
    plt.xlim([-0.5, len(names)-0.5])
    ylim = plt.gca().get_ylim()
    plt.ylim(ylim[0]-0.5e6, ylim[1])
#     ymin, ymax = np.min(hist_data), np.max(hist_data)
#     ymax -= 1e6
#     margin = 0.1*(ymax - ymin)
#     plt.ylim([ymin - margin, ymax + margin*2])
    # plt.yticks(np.linspace(275, 310, 8), np.linspace(275, 310, 8, dtype=int))
    plt.xlabel(x_label)


def plot_double_violin(names,  hist_data1, hist_data2, x_label='', y_label='', text_size=18):
    """@:param hist_data: a list of lists, for each name a list of metric values. """
    performance_data = []
    for i_arch in range(len(names)):
        for ret in hist_data1[i_arch]:
            performance_data.append((names[i_arch], means1[i_arch], ret))

    performance_df1 = pd.DataFrame(performance_data, columns=[x_label, 'Stability', 'stability'])

    performance_data2 = []
    for i_arch in range(len(names)):
        for ret in hist_data2[i_arch]:
            performance_data2.append((names[i_arch], means2[i_arch], ret))

    performance_df2 = pd.DataFrame(performance_data, columns=[x_label, 'Imitation', 'imitation'])

    performance_df = pd.concat([performance_df1, performance_df2])
    # plot violins
    violin = sns.violinplot(x=x_label, y='stability', hue='imitation', data=performance_df, inner='stick',
                            bw=0.5, width=1.25, linewidth=1.5)

    return True
    # plot horizontal lines indicating the means
    for i in range(len(means)):
        color = sns.color_palette()[i]
        plt.plot(np.arange(-1, len(names) + 1), np.ones((len(names) + 2,)) * means[i], c=color,
                 linestyle='--', linewidth=2, zorder=0)

        vertical_offset = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:-5.5e5}[i]
        plt.gca().text(-0.45, means[i]+1e5+vertical_offset,
                       f'{np.round(means[i]/1e6, 1)}M',
                       fontsize=text_size-1, color=color)

    # plot means
    plt.scatter(names, means, linestyle='None', color='#ffffff', s=40, zorder=10)
    plt.xlim([-0.5, len(names)-0.5])
#     ymin, ymax = np.min(hist_data), np.max(hist_data)
#     ymax -= 1e6
#     margin = 0.1*(ymax - ymin)
#     plt.ylim([ymin - margin, ymax + margin*2])
    # plt.yticks(np.linspace(275, 310, 8), np.linspace(275, 310, 8, dtype=int))
    plt.xlabel(x_label)


if __name__ == '__main__':
    norm_deltas = np.array([5209568, 5209664, 5810848, 6612304])/1e6
    bsln = np.array([16000000, 16000000, 16000000, 16000000])/1e6
    norm_angs = np.array([9417888, 9517888, 9000000, 9900000])/1e6
    aps = [bsln, norm_angs, norm_deltas]
    means = [np.mean(ap) for ap in aps]

    plot_violin(['Baseline', 'Normalized\nAngles', 'Normalized\nAngle Deltas'],
                means, aps, '', 'Training Timesteps [$x10^6$]')
    plt.show()