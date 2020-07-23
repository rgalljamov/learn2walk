import numpy as np
from os.path import dirname
from scripts.common import utils

def s(input):
    """ improves conversion of digits to strings """
    if isinstance(input, list):
        str_list = [str(item) for item in input]
        res = ''.join(str_list)
        return res
    return str(input)

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
    if (is_mod(MOD_BOUND_MEAN) or is_mod(MOD_SAC_ACTS)) and not is_mod(MOD_CUSTOM_NETS):
        raise TypeError("Using sac and tanh actions is only possible in combination"
                        "with the custom policy: MOD_CUSTOM_NETS.")

def is_mod(mod_str):
    return mod_str in modification

def do_run():
    return AP_RUN in approach

# choose approach
AP_DEEPMIMIC = 'dmm'
AP_RUN = 'run'
approach = AP_DEEPMIMIC

# choose modification
MOD_FLY = 'fly'
MOD_ORIG = 'orig'
MOD_PHASE_VAR = 'phase_var'

MOD_REFS_CONST = 'refs_const'
MOD_REFS_RAMP = 'refs_ramp'

MOD_CUSTOM_NETS = 'cstm_pi'
MOD_REW_MULT = 'rew_mult'
modification = mod([MOD_REW_MULT])
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
# load pretrained policy (behavior cloning)
MOD_PRETRAIN_PI = 'pretrain_pi'
# checking if learning is possible with weaker motors too
MOD_MAX_TORQUE = 'max_torque'
MAX_TORQUE = 300

modification = mod([MOD_CUSTOM_NETS, MOD_PRETRAIN_PI,
                    MOD_PI_OUT_DELTAS, MOD_NORM_ACTS])
assert_mod_compatibility()

# ----------------------------------------------------------------------------------
# Weights and Biases
# ----------------------------------------------------------------------------------
DEBUG = False
MAX_DEBUG_STEPS = int(2e4) # stop training thereafter!
logstd = 0
wb_project_name = 'behavior_clone'
wb_run_name = f'load FULL pi, obs_rms 200, logstd {s(logstd)}, ent-002'
wb_run_notes = 'Call wandb_init from cfg. fixed scope and weight loading: load weights without model! - Wrongly built the model with pretrained weights so far. Fixed it. |' \
               f'Set logstd to {s(logstd)} but add entropy bonus to increase exploration!' \
               'Load pretrained weights for the policy! THIS TIME ALSO THE OUTPUT LAYER!' \
               'const speed trajecs 400Hz | initializing obs_rms from previous run'
# ----------------------------------------------------------------------------------

# choose environment
envs = ['MimicWalker2d-v0', 'Walker2d-v2', 'Walker2d-v3', 'Humanoid-v3', 'Blind-BipedalWalker-v2', 'BipedalWalker-v2']
env_names = ['mim2d', 'walker2dv2', 'walker2dv3', 'humanoid', 'blind_walker', 'walker']
env_index = 0
env_id = envs[env_index]
env_name = env_names[env_index]

# choose hyperparams
algo = 'ppo2'
mio_steps = 4
n_envs = 16 if utils.is_remote() and not DEBUG else 1
batch_size = 8192 if utils.is_remote() else 1024
hid_layer_sizes = [128, 128]
lr_start = 1500
lr_final = 1125 # 1125 after 4M, 937.5 after 6M steps, should be 0 after 16M steps
cliprange = 0.15
ent_coef = -0.002
gamma = 0.99
_ep_dur_in_k = 4
ep_dur_max = int(_ep_dur_in_k * 1e3)

own_hypers = ''
info = ''
run_id = s(np.random.random_integers(0, 1000))

info_baseline_hyp_tune = f'hl{s(hid_layer_sizes)}_ent{int(ent_coef * 1000)}_lr{lr_start}to{lr_final}_epdur{_ep_dur_in_k}_' \
       f'bs{int(batch_size/1000)}_imrew6121_gam{int(gamma*1e3)}'

# construct the paths
abs_project_path = dirname(dirname(dirname(__file__))) + '/'
_mod_path = ('debug/' if DEBUG else '') + \
            f'{approach}/{modification}/{env_name}/{n_envs}envs/' \
            f'{algo}/{mio_steps}mio/'
hyp_path = (f'{own_hypers + info}/' if len(own_hypers + info) > 0 else '')
save_path_norun= abs_project_path + 'models/' + _mod_path + hyp_path
save_path = save_path_norun + f'{run_id}/'
init_obs_rms_path = abs_project_path + 'scripts/behavior_cloning/models/rms/env_999'

if DEBUG: print('Debugging model: ', save_path)
print('Modification:', modification)

# wandb
def get_wb_run_name():
    return wb_run_name + hyp_path + ' - ' + run_id
if len(wb_project_name) == 0:
    wb_project_name = _mod_path.replace('/', '_')[:-1]

# names of saved model before and after training
init_checkpoint = s(0)
final_checkpoint = s(999)


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