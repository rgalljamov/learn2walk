# -----------------------------
# Experiment Specification
# -----------------------------

# don't sync with W&B in debug mode, log additional information etc.
DEBUG_TRAINING = False
# maximum walking distance after which the episode is terminated
MAX_WALKING_DISTANCE = 22
# maximum length of an episode
MAX_EPISODE_STEPS = 3000

# TODO: remove COM reward, train longer with smaller LR decay, use exp clip_range sched
# configure Weights & Biases
WB_PROJECT_NAME = 'real_motors'
WB_EXPERIMENT_NAME = '300Nm, 4M, optimized foot-ground-contact initialization' # '140cm, REFS_RAMP, 8M, same lr decay, Rew 8200, 150Nm & 20Nm'
WB_EXPERIMENT_DESCRIPTION = 'Testing new initialization implementation' #'Walking slowly with the small walker. Had high variance between the runs before.'

# -----------------------------
# Simulation Environment
# -----------------------------

# the registered gym environment id, e.g. 'Walker2d-v2'
ENV_ID = 'MimicWalker3d-v0'
# walker XML file
WALKER_MJC_XML_FILE = 'walker3d_flat_feet.xml' # 'walker3d_flat_feet_lowmass.xml' # 'walker3d_flat_feet_40kg_140cm.xml' #
# simulation frequency... overwrite the frequency specified in the xml file
SIM_FREQ = 1000
# control frequency in Hz
CTRL_FREQ = 200
# does the model uses joint torques (True) or target angles (False)?
ENV_OUT_TORQUE = True
# peak joint torques [hip_sag, hip_front, knee_sag, ank_sag], same for both sides
PEAK_JOINT_TORQUES = [300]*4 # [150, 150, 150, 20] # [300, 300, 300, 300] #


# -----------------------------
# Algorithm Hyperparameters
# -----------------------------

# number of training steps = samples to collect [in Millions]
MIO_SAMPLES = 4
# how many parallel environments should be used to collect samples
N_PARALLEL_ENVS = 8
# network hidden layer sizes
hid_layer_sizes_vf = [512]*2
hid_layer_sizes_pi = [512]*2
# LR decay slope scaling: slope = lr_scale * (lr_final - lr_start)
# the decay is linear from lr_start to lr_final
lr_scale = 1