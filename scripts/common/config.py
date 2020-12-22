# suppress the annoying TF Warnings at startup
import warnings, sys
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# workaround to start scripts from cmd on remote server
sys.path.append('/home/rustam/code/remote/')

import numpy as np
from scripts.common import utils
from scripts import config_light as cfgl

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

def get_torque_ranges(hip_sag, hip_front, knee, ankle):
    torque_ranges = np.ones((8,2))
    peaks = np.array([hip_sag, hip_front, knee, ankle] * 2)
    torque_ranges[:,0] = -peaks
    torque_ranges[:,1] = peaks
    # print('Torque ranges (hip_sag, hip_front, knee, ankle): ', torque_ranges)
    return torque_ranges

def is_mod(mod_str):
    return mod_str in modification

# get the absolute path of the current project
abs_project_path = utils.get_absolute_project_path()

# approaches
AP_DEEPMIMIC = 'dmm'

# modifications / modes of the approach
MOD_FLY = 'fly'
MOD_ORIG = 'orig'

MOD_REFS_CONST = 'refs_const'
MOD_REFS_RAMP = 'refs_ramp'

MOD_CUSTOM_POLICY = 'cstm_pi'
MOD_REW_MULT = 'rew_mult'
# let the policy output deltas to current angle
MOD_PI_OUT_DELTAS = 'pi_deltas'

# use a tanh activation function at the output layer
MOD_BOUND_MEAN = 'tanh_mean'
# bound actions as done in SAC: apply a tanh to sampled actions
# and consider that squashing in the prob distribution, e.g. logpi calculation
MOD_SAC_ACTS = 'sac_acts'
# use running statistics from previous runs
MOD_LOAD_OBS_RMS = 'obs_rms'
init_obs_rms_path = abs_project_path + 'models/behav_clone/models/rms/env_999'
# load pretrained policy (behavior cloning)
MOD_PRETRAIN_PI = 'pretrain_pi'
# init the weights in the output layer of the value function to all zeros
MOD_VF_ZERO = 'vf_zero'
# checking if learning is possible with weaker motors too
MOD_MAX_TORQUE = 'max_torque'
TORQUE_RANGES = get_torque_ranges(*cfgl.PEAK_JOINT_TORQUES)


# mirror experiences
MOD_MIRROR_EXPS = 'mirr_exps'
# query the policy and the value functions to get neglogpacs and values
MOD_QUERY_NETS = 'query_nets'
# improve reward function by normalizing individual joints etc.
MOD_IMPROVE_REW = 'improve_rew'
# use linear instead of exponential reward to have better gradient away from trajecs
MOD_LIN_REW = 'lin_rew'
# use com x velocity instead of x position for com reward
MOD_COM_X_VEL = 'com_x_vel'
# use reference trajectories as a replay buffer
MOD_REFS_REPLAY = 'ref_replay'

# train multiple networks for different phases (left/right step, double stance)
MOD_GROUND_CONTACT_NNS = 'grnd_contact_nns'
MOD_3_PHASES = '3_phases'
MOD_CLIPRANGE_SCHED = 'clip_sched'
# use symmetrized mocap data for imitation reward
MOD_SYMMETRIC_WALK = 'sym_walk'
# reduce input dimensionality with an end-to-end encoder network of the observations
# e2e means here that we don't separately train the encoder to reconstruct the observations
MOD_E2E_ENC_OBS = 'e2e_enc_obs'
MOD_L2_REG = 'l2_reg'
l2_coef = 5e-4
# set a fixed logstd of the policy
MOD_CONST_EXPLORE = 'const_explor'
# learn policy for right step only, mirror states and actions for the left step
MOD_MIRR_PHASE = 'mirr_phase'
# TODO: Will this still be possible with domain randomization?
MOD_QUERY_VF_ONLY = 'query_vf_only'
MOD_REW_DELTA = 'rew_delta'
rew_delta_scale = 20
MOD_EXP_REPLAY = 'exp_replay'
replay_buf_size = 1

# only when training to accelerate with the velocity ramp trajectories
SKIP_N_STEPS = 1
STEPS_PER_VEL = 1

MOD_40KG = 'half_weight_40kg'
MOD_140cm_40KG = '140cm_40kg'
MAX_TORQUE = 300

# ------------------
approach = AP_DEEPMIMIC
SIM_FREQ = cfgl.SIM_FREQ
CTRL_FREQ = cfgl.CTRL_FREQ
# DO NOT CHANGE default modifications
modification = MOD_CUSTOM_POLICY + '/'
# HERE modifications can be added
modification += mod([MOD_MIRROR_EXPS])

# ----------------------------------------------------------------------------------
# Weights and Biases
# ----------------------------------------------------------------------------------
DEBUG = False or not sys.gettrace() is None or not utils.is_remote()
MAX_DEBUG_STEPS = int(2e4) # stop training thereafter!

rew_weights = '8110' if not is_mod(MOD_FLY) else '7300'
ent_coef = {200: -0.0075, 400: -0.00375}[CTRL_FREQ]
init_logstd = -0.7
pi_out_init_scale = 0.001
cliprange = 0.15
clip_start = 0.55 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_end = 0.1 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_exp_slope = 5

enc_layer_sizes = [512]*2 + [16]
hid_layer_sizes_vf = [512]*2
hid_layer_sizes_pi = [512]*2
gamma = {50:0.99, 100: 0.99, 200:0.995, 400:0.998}[CTRL_FREQ]
rew_scale = 1
et_rew_thres = 0.1 * rew_scale
alive_bonus = 0.2 * rew_scale
# number of episodes per model evaluation
EVAL_N_TIMES = 20
# num of times a batch of experiences is used
noptepochs = 4

wb_project_name = 'cleanup'
wb_run_name = ('SYM ' if is_mod(MOD_SYMMETRIC_WALK) else '') + \
               'CC4.4: ALIVE bonus, REALLY pre commit, more strict angET'
wb_run_notes = f'added alive bonus back, was removed accidentally! ' \
               f'changed ET to the more strict angle ranges, just changed com z pos to 0.75' \
               f'Refactored ET rew calculation and solved smaller issues.' \
               f'Made ET a bit more strict again, retained min com z pos at 0.5. Refactored mimic and derived classes.' \
               f'CC4.2: old ET (weakened) and old ET rew calculation + Changed what class MimicEnv is derived from! ' \
               f'Weaken ET by increasing allowed angle deviations and lowering min COM Z Pos.' \
               f'CC4.2 added ET based on angles again, as otherwise Mujoco Exception! ' \
               f'CC4: removed ET based on angles and simplified ET reward calculation.' \
               f'CC3: make action normalization default and remove mode; set init flag to True. ' \
               f'CC2.1: Set mimic_env.inited flag to False before starting evaluation! ' \
               f'CC2.0: Removed check if refs are None at the beginning of step(). ' \
               f'CC1.5:Removed possibility to add ground contact information! ' \
               f'CC1: removed multiple properties and methods. Added assertions. ' \
               f'Use gear ratio 1 and scale actions by MAX_TORQUE in the environment. ' \
               f'Repeat Baseline experiment with original walker model.'

# ----------------------------------------------------------------------------------

# choose environment
if cfgl.ENV_ID is not None:
    env_id = cfgl.ENV_ID
    # short description of the env used in the save path
    env_abbrev = env_id
    env_is3d = cfgl.ENV_IS_3D
    env_out_torque = cfgl.ENV_OUT_TORQUE
else:
    env_ids = ['MimicWalker2d-v0', 'MimicWalker2d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'Walker2d-v2', 'Walker2d-v3', 'Humanoid-v3', 'Blind-BipedalWalker-v2', 'BipedalWalker-v2']
    env_abbrevs = ['mim2d', 'mim_trq2d', 'mim3d', 'mim_trq3d', 'mim_trq_ff3d', 'walker2dv2', 'walker2dv3', 'humanoid', 'blind_walker', 'walker']
    env_index = 4
    env_id = env_ids[env_index]
    # used in the save path (e.g. 'wlk2d')
    env_abbrev = env_abbrevs[env_index]
    env_out_torque = True
    env_is3d = True


# choose hyperparams
algo = 'ppo2'
# reward the agent gets when max episode length was reached
ep_end_reward = 10
# reward for an early terminal state
et_reward = -100
# number of experiences to collect, not training steps.
# In case of mirroring, during 4M training steps, we collect 8M samples.
mirr_exps = is_mod(MOD_MIRROR_EXPS)
exp_replay = is_mod(MOD_EXP_REPLAY)
mio_samples = 4
if mirr_exps: mio_samples *= 2
n_envs = 8 if utils.is_remote() and not DEBUG else 2
minibatch_size = 512 * 4
batch_size = (4096 * 4 * (2 if not mirr_exps else 1)) if not DEBUG else 2*minibatch_size
# to make PHASE based mirroring comparable with DUP, reduce the batch size
if is_mod(MOD_MIRR_PHASE): batch_size = int(batch_size / 2)
# if using a replay buffer, we have to collect less experiences
# to reach the same batch size
if exp_replay: batch_size = int(batch_size/(replay_buf_size+1))

lr_start = 500
lr_final = 1
_ep_dur_in_k = {400: 6, 200: 3, 100: 1.5, 50: 0.75}[CTRL_FREQ]
ep_dur_max = int(_ep_dur_in_k * 1e3)
max_distance = 22

run_id = s(np.random.random_integers(0, 1000))
info_baseline_hyp_tune = f'hl{s(hid_layer_sizes_vf)}_ent{int(ent_coef * 1000)}_lr{lr_start}to{lr_final}_epdur{_ep_dur_in_k}_' \
       f'bs{int(batch_size/1000)}_imrew{rew_weights}_gam{int(gamma*1e3)}'

# construct the paths to store the models at
_mod_path = ('debug/' if DEBUG else '') + \
            f'{approach}/{modification}/{env_abbrev}/{n_envs}envs/' \
            f'{algo}/{mio_samples}mio/'
save_path_norun= abs_project_path + 'models/' + _mod_path
save_path = save_path_norun + f'{run_id}/'

print('Model: ', save_path)
print('Modification:', modification)

# wandb
def get_wb_run_name():
    return wb_run_name

# names of saved model before and after training
init_checkpoint = 'init'
final_checkpoint = 'final'

if __name__ == '__main__':
    from scripts.train import train
    train()