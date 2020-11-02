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
        print("Normalized actions (ctrlrange [-1,1] for all joints) " \
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
MOD_TORQUE_DELTAS = 'trq_delta'
MOD_L2_REG = 'l2_reg'
# set a fixed logstd of the policy
MOD_CONST_EXPLORE = 'const_explor'
# learn policy for right step only, mirror states and actions for the left step
MOD_MIRR_STEPS = 'steps_mirr'
MOD_QUERY_VF_ONLY = 'query_vf_only'
MOD_REW_DELTA = 'rew_delta'
MOD_EXP_REPLAY = 'exp_replay'
replay_buf_size = 1
MOD_N_OPT_EPS_SCHED = 'opt_eps_sched'

# ------------------
approach = AP_DEEPMIMIC
CTRL_FREQ = 200
modification = mod([MOD_NORM_ACTS,
    MOD_CUSTOM_POLICY,
    ])
assert_mod_compatibility()

# ----------------------------------------------------------------------------------
# Weights and Biases
# ----------------------------------------------------------------------------------
DEBUG = False or not sys.gettrace() is None or not utils.is_remote()
MAX_DEBUG_STEPS = int(2e4) # stop training thereafter!

rew_weights = '8110' if not is_mod(MOD_FLY) else '7300'
ent_coef = {200: -0.0075, 400: -0.00375}[CTRL_FREQ]
# if is_mod(MOD_MIRROR_EXPS): ent_coef /= 2
# if is_mod(MOD_EXP_REPLAY): ent_coef /= (replay_buf_size+1)
init_logstd = -0.7
pi_out_init_scale = 0.001
cliprange = 0.15
clip_start = 0.55 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_end = 0.1 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_exp_slope = 5

opt_eps_start = 11
opt_eps_end = 4
opt_eps_slope = 10

SKIP_N_STEPS = 1
STEPS_PER_VEL = 1
enc_layer_sizes = [512]*2 + [16]
hid_layer_sizes_vf = [512]*2
hid_layer_sizes_pi = [512]*2
gamma = {50:0.99, 100: 0.999, 200:0.995, 400:0.998}[CTRL_FREQ]
alive_min_dist = 0
trq_delta = 0.25
rew_scale = 1
l2_coef = 5e-4
et_rew_thres = 0.1 * rew_scale
alive_bonus = 0.2 * rew_scale
EVAL_N_TIMES = 20
rew_delta_scale = 20

wb_project_name = 'mrr_phase3d'
wb_run_name = ('SYM ' if is_mod(MOD_SYMMETRIC_WALK) else '') + \
               f'MRR steps, half BS'
               # f'exp clip decay (VF too): {clip_start} - {clip_end}'
               # f'PI E2ENC {enc_layer_sizes}, pi {hid_layer_sizes_pi[0]}'
               # f'exp noptepochs schedule: slope {opt_eps_slope}, {opt_eps_start} - {opt_eps_end}'
               # f'Replay BUF{replay_buf_size}, retain BS, ent_coef{ent_coef}, query both, delete pacs'
               # f'MRR no query, init logstd {init_logstd}, half ent_coef{ent_coef}'
wb_run_notes = f'' \
               'Changed evaluation of stable walks to consider 18m without falling as stable. '\
               'Evaluate the agent starting at 75% of the step cycle. ' \
               'Removed reward scaling! Reduced episode duration to 3k instead of 3.2k; ' \
               'Increased the minimum learning rate to 1e-6, was -8 before. ' \
               'Reduced the minimum logstd to -2.3, ' \
               'Estimate mean return of an action based on gamma and mean ep rew. ' \
               'Get a big positive reward on episode end to avoid punishing good actions ' \
               'at the end of the episode due to small return. ' \
               'Flat feet. ' \
               'extended epdur to have enough time to reach episode end and stop episode only after 22m. ' \
               'softened ET by allowing more trunk rotation in the axial plane axial_dev0.5. ' \
               '' \
               'no longer stop the episode based on a minimum reward signal! ' \
               '' \
               'Added hard ET conditions to avoid falling: ' \
               'trunk angles are checked in all three directions, ' \
               'com height has much higher threshold!' \
               '' \
               '' \
               ' ' \
               'Eval model from 20 different steps at same position. '
# num of times a batch of experiences is used
if is_mod(MOD_N_OPT_EPS_SCHED):
    noptepochs = f'{opt_eps_start} - {opt_eps_end}'
    # schedule is setup in train.py
    opt_eps_sched = None
else:
    noptepochs = 4

# ----------------------------------------------------------------------------------

# choose environment
envs = ['MimicWalker2d-v0', 'MimicWalker2d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'Walker2d-v2', 'Walker2d-v3', 'Humanoid-v3', 'Blind-BipedalWalker-v2', 'BipedalWalker-v2']
env_names = ['mim2d', 'mim_trq2d', 'mim3d', 'mim_trq3d', 'mim_trq_ff3d', 'walker2dv2', 'walker2dv3', 'humanoid', 'blind_walker', 'walker']
env_index = 4
env_id = envs[env_index]
env_name = env_names[env_index]
is_torque_model = env_name in ['mim_trq2d', 'mim_trq3d', 'mim_trq_ff3d', 'walker2dv2', 'walker2dv3', 'humanoid', 'blind_walker', 'walker']

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
mio_steps = (8 if is_torque_model else 16) * (2 if mirr_exps else 1)
n_envs = 8 if utils.is_remote() and not DEBUG else 2
minibatch_size = 512 * 4
batch_size = (4096 * 4 * (2 if not mirr_exps else 1)) if not DEBUG else 2*minibatch_size
if is_mod(MOD_MIRR_STEPS): batch_size = int(batch_size/2)
# when mirroring experiences, we have to duplicate the number of minibatches
# otherwise only half of the data will be used (see implementation of PPO2 updates)
# todo: remove, as n_minibatches is automatically calculated in CustomPPO2
#       to maintain the same batch and minibatch sizes.
n_mini_batches = int(batch_size / minibatch_size) * (2 if mirr_exps else 1)
# if using a replay buffer, we have to collect less experiences
# to reach the same batch size
if exp_replay: batch_size = int(batch_size/(replay_buf_size+1))


lr_start = 2000 if is_mod(MOD_EXP_LR_SCHED) else 500
mio_steps_to_lr1 = 16 # (32 if is_mod(MOD_MIRROR_EXPS) else 16)
slope = mio_steps/mio_steps_to_lr1
lr_final = 50 if is_mod(MOD_EXP_LR_SCHED) else 1
_ep_dur_in_k = {400: 6, 200: 3, 100: 1.5, 50: 0.75}[CTRL_FREQ]
ep_dur_max = int(_ep_dur_in_k * 1e3)
max_distance = 22

own_hypers = ''
info = ''
run_id = s(np.random.random_integers(0, 1000))

info_baseline_hyp_tune = f'hl{s(hid_layer_sizes_vf)}_ent{int(ent_coef * 1000)}_lr{lr_start}to{lr_final}_epdur{_ep_dur_in_k}_' \
       f'bs{int(batch_size/1000)}_imrew{rew_weights}_gam{int(gamma*1e3)}'

# construct the paths
abs_project_path = utils.get_absolute_project_path()
_mod_path = ('debug/' if DEBUG else '') + \
            f'{approach}/{modification}/{env_name}/{n_envs}envs/' \
            f'{algo}/{mio_steps}mio/'
hyp_path = (f'{own_hypers + info}/' if len(own_hypers + info) > 0 else '')
save_path_norun= abs_project_path + 'models/' + _mod_path + hyp_path
save_path = save_path_norun + f'{run_id}/'
init_obs_rms_path = abs_project_path + 'models/behav_clone/models/rms/env_999'
if is_mod(MOD_FLY):
    init_obs_rms_path = abs_project_path + 'models/behav_clone/models/' \
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