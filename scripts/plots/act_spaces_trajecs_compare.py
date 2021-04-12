import numpy as np
import seaborn as sns
from scripts.common.utils import config_pyplot, change_plot_properties

plt = config_pyplot((9.6, 6.4))
font_size, tick_size, legend_size = \
        change_plot_properties(font_size=-1, tick_size=1, line_width=+1)
sns.set_style('ticks')

ANGLE = 0
DELTA = 1
TORQUE = 2

### LOAD ALL TRAJECTORIES
path = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/scripts/plots/act_spaces_trajecs_data/'
files = 'angles, deltas, torque'.split(', ')

# shape of saved trajecs is (4:sim, ref, mean, std; 8:4xpos and 4vels; 2000:points)
sim_data = np.empty((6, 2000))
ref_data = np.empty((6, 2000))

# choose range of data to display
x_min, x_max = 900, 1412

for i, file in enumerate(files):
    data = np.load(path + file + '.npy')
    sim_data[i, :] = data[0, 2, :]
    sim_data[i + 3, :] = data[0, 6, :]
    ref_data[i, :] = data[1, 2, :]
    ref_data[i + 3, :] = data[1, 6, :]

fig, subs = plt.subplots(2, 3, sharex=True, sharey='row')
for i, sub in enumerate(subs.flatten()):
    sub.plot(sim_data[i, x_min+(140 if i==0 else 0):x_max+(140 if i==0 else 0)], alpha=1, zorder=10)
    sub.plot(ref_data[i, x_min+(140 if i==0 else 0):x_max+(140 if i==0 else 0)], linewidth=6, alpha=0.75)
    sns.despine()

    if i<3:
        sub.set_xticks([])
    else:
        # xticks
        x_range = x_max - x_min
        # show 0 - 2.4 seconds for the ticks
        range24 = int(2.4/(512/200) * x_range)
        xticks = np.linspace(0, range24, 4)
        sub.set_xticks(xticks)
        sub.set_xticklabels(np.round(xticks/200,1))
    sub.set_xlabel('Time [s]')

# lims
subs[1,0].set_ylim([-12.5,12.5])
subs[0,0].set_xlim([0, x_range])

# labels
subs[0,0].set_ylabel('Position [rad]')
subs[1,0].set_ylabel('Velocity [rad/s]')


y_title = 1.2
subs[0,0].text(0.5, y_title, '(a) Angle', fontsize=font_size, weight='bold',
                 horizontalalignment='center', verticalalignment='center',
                 transform=subs[0,0].transAxes)
subs[0,1].text(0.5, y_title, '(b) Angle Delta', fontsize=font_size, weight='bold',
                 horizontalalignment='center', verticalalignment='center',
                 transform=subs[0,1].transAxes)
subs[0,2].text(0.5, y_title, '(c) Torque', fontsize=font_size, weight='bold',
                 horizontalalignment='center', verticalalignment='center',
                 transform=subs[0,2].transAxes)

fig.align_ylabels()

# legend
subs[1,2].legend(['Model', 'Human'], fancybox=True, framealpha=0.6,
                 loc='lower center', handlelength=0.5,
                 fontsize=font_size-6, ncol=2,
                 frameon=False)
plt.show()
