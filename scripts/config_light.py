# -----------------------------
# Experiment Specification
# -----------------------------

# don't sync with W&B in debug mode, log additional information etc.
DEBUG_TRAINING = False
# maximum walking distance after which the episode is terminated
MAX_WALKING_DISTANCE = 22
# maximum length of an episode
MAX_EPISODE_STEPS = 3000

# configure Weights & Biases
WB_PROJECT_NAME = 'cleanup'
WB_EXPERIMENT_NAME = 'CC7: PRE COMMIT finishing config light'
WB_EXPERIMENT_DESCRIPTION = 'Added most important hypers to config light.'


# -----------------------------
# Simulation Environment
# -----------------------------

# the registered gym environment id, e.g. 'Walker2d-v2'
ENV_ID = 'MimicWalker3d-v0'
# simulation frequency... overwrite the frequency specified in the xml file
SIM_FREQ = 1000
# control frequency in Hz
CTRL_FREQ = 200
# does the model uses joint torques (True) or target angles (False)?
ENV_OUT_TORQUE = True
# peak joint torques [hip_sag, hip_front, knee_sag, ank_sag], same for both sides
PEAK_JOINT_TORQUES = [300, 300, 300, 300]


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