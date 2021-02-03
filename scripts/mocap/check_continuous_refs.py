import numpy as np
import scipy.io as spio
import seaborn as sns
from scripts.common import utils
from matplotlib import pyplot as plt
from scripts.common.config import abs_project_path

# plt = utils.config_pyplot(font_size=0, tick_size=0, legend_fontsize=0)
# plt.rcParams.update({'figure.autolayout': False})

# load matlab data, containing trajectories of 250 steps
dir_path = abs_project_path
file_path = 'assets/ref_trajecs/02_walking0001.mat'

data = spio.loadmat(dir_path+file_path, squeeze_me=True)

kin_labels = data['rowNameIK']
angles = data['angJoi']
ang_vels = data['angDJoi']
dofs = len(kin_labels)
SAMPLE_FREQ = 500 # fskin


test_refs = False
if test_refs:
    from scripts.mocap.ref_trajecs import ReferenceTrajectories as RT

    rt = RT(range(15), range(15,29))
    rt._step = rt.data[0]
    compos, comvel = rt.get_com_kinematics_full()
    step = rt._step
    dofs, timesteps = step.shape
    # step[0:3,:] -= compos


# label every trajectory with the corresponding name
labels = ['COM Pos (X)', 'COM Pos (Y)', 'COM Pos (Z)',
          'Trunk Rot (quat,w)', 'Trunk Rot (quat,x)', 'Trunk Rot (quat,y)', 'Trunk Rot (quat,z)',
          'Ang Hip Frontal R', 'Ang Hip Sagittal R',
          'Ang Knee R', 'Ang Ankle R',
          'Ang Hip Frontal L', 'Ang Hip Sagittal L',
          'Ang Knee L', 'Ang Ankle L',

          'COM Vel (X)', 'COM Vel (Y)', 'COM Vel (Z)',
          'Trunk Ang Vel (X)', 'Trunk Ang Vel (Y)', 'Trunk Ang Vel (Z)',
          'Vel Hip Frontal R', 'Vel Hip Sagittal R',
          'Vel Knee R', 'Vel Ankle R',
          'Vel Hip Frontal L', 'Vel Hip Sagittal L',
          'Vel Knee L', 'Vel Ankle L',

          'Foot Pos L (X)', 'Foot Pos L (Y)', 'Foot Pos L (Z)',
          'Foot Pos R (X)', 'Foot Pos R (Y)', 'Foot Pos R (Z)',

          'GRF R [N]', 'GRF L [N]',
          'Trunk Rot (euler,x)', 'Trunk Rot (euler,y)', 'Trunk Rot (euler,z)',
          ]

labels = kin_labels

# plot figure in full screen mode (scaled down aspect ratio of my screen)
plt.rcParams['figure.figsize'] = (19.2, 10.8)
plt.rcParams.update({'figure.autolayout': True})


for i in range(dofs):
    try: subplt = plt.subplot(8,5,i+1, sharex=subplt)
    except: subplt = plt.subplot(8,5,i+1)
    line_blue = plt.plot(angles[i, 1000:2000])
    velplt = subplt.twinx()
    line_orange = velplt.plot(ang_vels[i, 1000:2000], 'darkorange')
    velplt.tick_params(axis='y', labelcolor='darkorange')
    plt.title(f'{i} - {labels[i]}')

plt.rcParams.update({'figure.autolayout': True})



    # remove x labels from first rows
    # if i < 32:
    #     plt.xticks([])

# collect different lines to place the legend in a separate subplot
lines = [line_blue[0], line_orange[0]]
# plot the legend in a separate subplot
with sns.axes_style("white", {"axes.edgecolor": 'white'}):
    legend_subplot = plt.subplot(8, 5, i + 2)
    legend_subplot.set_xticks([])
    legend_subplot.set_yticks([])
    legend_subplot.legend(lines, ['Joint Positions [rad]',
                                  'Joint Position Derivatives [rad/s]',
                                  'Joint Velocities (Dataset) [rad/s]'],
                          bbox_to_anchor=(1.15, 1.05) )

# # fix title overlapping when tight_layout is true
# plt.gcf().tight_layout(rect=[0, 0, 1, 0.95])
# plt.subplots_adjust(wspace=0.55, hspace=0.5)
# plt.suptitle('Trajectories')

plt.show()
