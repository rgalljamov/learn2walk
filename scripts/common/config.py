# suppress the annoying TF Warnings at startup
import warnings, sys
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# workaround to start scripts from cmd on remote server
sys.path.append('/home/rustam/code/remote/')

import numpy as np
from scripts.common import utils

def s(input):
    """ improves conversion of digits to strings """
    if isinstance(input, list):
        str_list = [str(item) for item in input]
        res = ''.join(str_list)
        return res
    return str(input).replace('.','')

def mod(mods:list):
    modification = ''
    for mod in mods:
        modification += mod + '/'
    # remove last /
    modification = modification[:-1]
    return modification

def assert_mod_compatibility():
    """
    Some modes cannot be used together. In such cases,
    this function throws an exception and provides explanations.
    """
    if is_mod(MOD_NORM_ACTS) and not is_mod(MOD_PI_OUT_DELTAS):
        raise TypeError("Normalized actions (ctrlrange [-1,1] for all joints) " \
                        "currently only work when policy outputs delta angles.")
    if (is_mod(MOD_BOUND_MEAN) or is_mod(MOD_SAC_ACTS)) and not is_mod(MOD_CUSTOM_POLICY):
        raise TypeError("Using sac and tanh actions is only possible in combination"
                        "with the custom policy: MOD_CUSTOM_NETS.")

def get_torque_ranges(hip, knee, ankle):
    torque_ranges = np.ones((6,2))
    peaks = np.array([hip, knee, ankle] * 2)
    torque_ranges[:,0] = -peaks
    torque_ranges[:,1] = peaks
    # print('Torque ranges (hip, knee, ankle): ', (hip, knee, ankle))
    return torque_ranges

def is_mod(mod_str):
    return mod_str in modification

def do_run():
    return AP_RUN in approach

# choose approach
AP_DEEPMIMIC = 'dmm'
AP_RUN = 'run'
AP_BEHAV_CLONE = 'bcln'

# choose modification
MOD_FLY = 'fly'
MOD_ORIG = 'orig'
MOD_PHASE_VAR = 'phase_var'

MOD_REFS_CONST = 'refs_const'
MOD_REFS_RAMP = 'refs_ramp'

MOD_CUSTOM_POLICY = 'cstm_pi'
MOD_REW_MULT = 'rew_mult'
# allow the policy to output angles in the maximum range
# but punish actions that are too far away from current angle
MOD_PUNISH_UNREAL_TARGET_ANGS = 'pun_unreal_angs'
# let the policy output deltas to current angle
MOD_PI_OUT_DELTAS = 'pi_deltas'
# normalize actions: programmatically set action space to be [-1,1]
MOD_NORM_ACTS = 'norm_acts'
# init weights in the policy output layer to zero (action=qpos+pi_out)
MOD_ZERO_OUT = 'zero_out' # - not tried yet
# use a tanh activation function at the output layer
MOD_BOUND_MEAN = 'tanh_mean'
# bound actions as done in SAC: apply a tanh to sampled actions
# and consider that squashing in the prob distribution, e.g. logpi calculation
MOD_SAC_ACTS = 'sac_acts'
# use running statistics from previous runs
MOD_LOAD_OBS_RMS = 'obs_rms'
# load pretrained policy (behavior cloning)
MOD_PRETRAIN_PI = 'pretrain_pi'
# init the weights in the output layer of the value function to all zeros
MOD_VF_ZERO = 'vf_zero'
# checking if learning is possible with weaker motors too
MOD_MAX_TORQUE = 'max_torque'
TORQUE_RANGES = get_torque_ranges(300, 300, 300)
# Reduce dimensionality of the state with a pretrained encoder
MOD_ENC_DIM_RED_PRETRAINED = 'dim_red'
# use mocap statistics for ET
MOD_REF_STATS_ET = 'ref_et'
et_rew_thres = 0.1
# mirror experiences
MOD_MIRROR_EXPS = 'mirr_exps'
# improve reward function by normalizing individual joints etc.
MOD_IMPROVE_REW = 'improve_rew'
# use linear instead of exponential reward to have better gradient away from trajecs
MOD_LIN_REW = 'lin_rew'
# use com x velocity instead of x position for com reward
MOD_COM_X_VEL = 'com_x_vel'
# use reference trajectories as a replay buffer
MOD_REFS_REPLAY = 'ref_replay'
# train VF and PI on ref trajectories during the first policy update
MOD_ONLINE_CLONE = 'online_clone'
# input ground contact information
MOD_GROUND_CONTACT = 'grnd_contact'
# double stance results in 0, 0, 1
MOD_GRND_CONTACT_ONE_HOT = 'grnd_1hot'
# train multiple networks for different phases (left/right step, double stance)
MOD_GROUND_CONTACT_NNS = 'grnd_contact_nns'
MOD_3_PHASES = '3_phases'
MOD_CLIPRANGE_SCHED = 'clip_sched'
MOD_EXP_LR_SCHED = 'expLRdec'
MOD_SYMMETRIC_WALK = 'sym_walk'
# reduce input dimensionality with an end-to-end encoder network of the observations
# e2e means here that we don't separately train the encoder to reconstruct the observations
MOD_E2E_ENC_OBS = 'e2e_enc_obs'

# ------------------
approach = AP_DEEPMIMIC
modification = mod([MOD_E2E_ENC_OBS,
    MOD_CUSTOM_POLICY
    ])
assert_mod_compatibility()

# ----------------------------------------------------------------------------------
# Weights and Biases
# ----------------------------------------------------------------------------------
DEBUG = False or not sys.gettrace() is None or not utils.is_remote()
MAX_DEBUG_STEPS = int(2e4) # stop training thereafter!

rew_weights = '8110' if not is_mod(MOD_FLY) else '7300'
ent_coef = 0 # -0.005
logstd = 0
cliprange = 0.15
clip_start = 0.3 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_end = 0.1 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
SKIP_N_STEPS = 1
STEPS_PER_VEL = 1
enc_layer_sizes = [512]*2 + [16]
hid_layer_sizes = [512]*2


# add negative ent_coef to counteract growing entropy
wb_project_name = 'e2enc3d'
wb_run_name = f'{enc_layer_sizes + hid_layer_sizes}'
wb_run_notes = 'End to End Encoder reduces dim of the observations!' \
               'Normalized Actions: 1 -> 300Nm. ' \
               'Adjusted hypers from the 2D walker. ' \
               'Minibatch-Size is already set to 512, which was actually' \
               'an improvement learned later!'
# num of times a batch of experiences is used
noptepochs = 4

# ----------------------------------------------------------------------------------

# choose environment
envs = ['MimicWalker2d-v0', 'MimicWalker2d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'Walker2d-v2', 'Walker2d-v3', 'Humanoid-v3', 'Blind-BipedalWalker-v2', 'BipedalWalker-v2']
env_names = ['mim2d', 'mim_trq2d', 'mim3d', 'mim_trq3d', 'walker2dv2', 'walker2dv3', 'humanoid', 'blind_walker', 'walker']
env_index = 3
env_id = envs[env_index]
env_name = env_names[env_index]

# choose hyperparams
algo = 'ppo2'
# reward the agent gets when max episode length was reached
ep_end_reward = 10
# reward for an early terminal state
et_reward = -100
# number of experiences to collect, not training steps.
# In case of mirroring, during 4M training steps, we collect 8M samples.
mio_steps = 12 * (2 if is_mod(MOD_MIRROR_EXPS) else 1)
n_envs = 8 if utils.is_remote() and not DEBUG else 1
batch_size = (4096 * (2 if not is_mod(MOD_MIRROR_EXPS) else 1)) if not DEBUG else 1024
minibatch_size = 512
n_mini_batches = int(batch_size / minibatch_size)
lr_start = 1500 if is_mod(MOD_EXP_LR_SCHED) else 500
mio_steps_to_lr1 = 16 # (32 if is_mod(MOD_MIRROR_EXPS) else 16)
slope = mio_steps/mio_steps_to_lr1
lr_final = 0.001
gamma = 0.99
_ep_dur_in_k = 3
ep_dur_max = int(_ep_dur_in_k * 1e3) + 100

own_hypers = ''
info = ''
run_id = s(np.random.random_integers(0, 1000))

info_baseline_hyp_tune = f'hl{s(hid_layer_sizes)}_ent{int(ent_coef * 1000)}_lr{lr_start}to{lr_final}_epdur{_ep_dur_in_k}_' \
       f'bs{int(batch_size/1000)}_imrew{rew_weights}_gam{int(gamma*1e3)}'

# construct the paths
abs_project_path = utils.get_absolute_project_path()
_mod_path = ('debug/' if DEBUG else '') + \
            f'{approach}/{modification}/{env_name}/{n_envs}envs/' \
            f'{algo}/{mio_steps}mio/'
hyp_path = (f'{own_hypers + info}/' if len(own_hypers + info) > 0 else '')
save_path_norun= abs_project_path + 'models/' + _mod_path + hyp_path
save_path = save_path_norun + f'{run_id}/'
init_obs_rms_path = abs_project_path + 'scripts/behavior_cloning/models/rms/env_999'
if is_mod(MOD_FLY):
    init_obs_rms_path = abs_project_path + 'scripts/behavior_cloning/models/' \
                                           'rms/env_rms_fly_const_speed'

print('Model: ', save_path)
print('Modification:', modification)

# wandb
def get_wb_run_name():
    return wb_run_name + hyp_path # + ' - ' + run_id
if len(wb_project_name) == 0:
    wb_project_name = _mod_path.replace('/', '_')[:-1]

# names of saved model before and after training
init_checkpoint = 'init'
final_checkpoint = 'final'

if __name__ == '__main__':
    from scripts.train import train
    train()


'''
# ARCHS
ARC_FC_MLP = 'mlp'
ARC_ACTION_BRANCHING = 'act_brnchg'

# modifs
MOD_SUPERVISED_INIT = 'supervised_init'
MOD_ORIG_FC = 'orig_fc'
MOD_CSTM_MLP = 'cstm_mlp'
MOD_SHARE_ALL_HID = 'share_all_hids'

MOD_ATTENTION = 'attention'
MOD_SPARSE_W_INIT = 'sprs_w_init'
MOD_L1_REG = 'l1_reg'

AT_RELU = 'at_relu'
AT_SIGMOID = 'at_sigm'
AT_LEARNABLE_PARAMS = 'learn_params'
AT_REMOID = 'at_remoid'
AT_SOFTMAX = 'at_softmax'

# maybe useful later
hid_size = 128
l1_scale = 5e-2
zero_prct = 0.70
rem_slope = 4
rem_prct_init = 0.5
sftmx_scale = 8
sftmx_init = 0.01
sftmx_shift = 0
sftm_inv_temp = 1
'''