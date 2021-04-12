# -----------------------------
# Experiment Specification
# -----------------------------

# don't sync with W&B in debug mode, log additional information etc.
DEBUG_TRAINING = False
# maximum walking distance after which the episode is terminated
MAX_WALKING_DISTANCE = 10
# maximum length of an episode
MAX_EPISODE_STEPS = 3000

# configure Weights & Biases
WB_PROJECT_NAME = 'train_dur_comparison'
WB_EXPERIMENT_NAME = 'PD normalized - 8Mio steps'
WB_EXPERIMENT_DESCRIPTION = 'Checking if the superiority of the torque model ' \
                            'compared to the PD model is due to the shorter training time' \
                            'which changes the learning rate schedule'

# -----------------------------
# Simulation Environment
# -----------------------------

# the registered gym environment id, e.g. 'Walker2d-v2'
ENV_ID = 'MimicWalker3d-v0'
# walker XML file
WALKER_MJC_XML_FILE = 'walker3d_flat_feet.xml' #'walker3d_flat_feet_hip3d.xml' # 'walker3d_flat_feet_lowmass.xml' # 'walker3d_flat_feet_40kg_140cm.xml' #
# simulation frequency... overwrite the frequency specified in the xml file
IS_HIP_3D = False
SIM_FREQ = 1000
# control frequency in Hz
CTRL_FREQ = 200
# does the model uses joint torques (True) or target angles (False)?
ENV_OUT_TORQUE = True
# peak joint torques [hip_sag, hip_front, knee_sag, ank_sag], same for both sides
PEAK_JOINT_TORQUES = [300, 300, 300, 300] # [300]*4 # [50]*3 + [5] #


# -----------------------------
# Algorithm Hyperparameters
# -----------------------------

# number of training steps = samples to collect [in Millions]
MIO_SAMPLES = 8
# how many parallel environments should be used to collect samples
N_PARALLEL_ENVS = 8
# network hidden layer sizes
hid_layer_sizes_vf = [512]*2
hid_layer_sizes_pi = [512]*2
# LR decay slope scaling: slope = lr_scale * (lr_final - lr_start)
# the decay is linear from lr_start to lr_final
lr_scale = 1