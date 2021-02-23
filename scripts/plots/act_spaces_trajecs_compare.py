import numpy as np
import seaborn as sns
from scripts.common import utils

plt = utils.import_pyplot()
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

for i, file in enumerate(files):
    data = np.load(path + file + '.npy')
    sim_data[i, :] = data[0, 2, :]
    sim_data[i + 3, :] = data[0, 6, :]
    ref_data[i, :] = data[1, 2, :]
    ref_data[i + 3, :] = data[1, 6, :]

fig, subs = plt.subplots(2, 3, sharex=True)
for i in range(6):
    subs.flatten()[i].plot(sim_data[i, :])
    subs.flatten()[i].plot(ref_data[i, :])

plt.show()
