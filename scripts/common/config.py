import numpy as np
from os.path import dirname, abspath

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


def s(input):
    """ improves conversion of digits to strings """
    return str(input).replace('.','')

def mod(mods:list):
    modification = ''
    for mod in mods:
        modification += mod + '/'
    # remove last /
    modification = modification[:-1]
    return modification

# choose approach
AP_DEEPMIMIC = 'deepmim'
AP_RUN = 'run'
approach = AP_DEEPMIMIC

MOD_FLY = 'fly'
MOD_ORIG = 'orig'
MOD_PHASE_VAR = 'phase_var'
MOD_REAL_TORQUE_PEAKS = 'real_torque'
MOD_TORQUE_500 = '500Nm'
# no phase variable, minimal state/action spaces, weak ET, no endeffector reward
MOD_MINIMAL = 'minimal'
modification = MOD_REAL_TORQUE_PEAKS # mod([MOD_MINIMAL, MOD_REW_ET])

# config environment
n_parallel_envs = 1
envs = ['MimicWalker2d-v0', 'Walker2d-v2', 'Walker2d-v3', 'Humanoid-v3', 'Blind-BipedalWalker-v2', 'BipedalWalker-v2']
env_names = ['mim_walker2d', 'walker2dv2', 'walker2dv3', 'humanoid', 'blind_walker', 'walker']
env_index = 0
env_id = envs[env_index]
env_name = env_names[env_index]

# default hyperparameters from stable-baselines
HYPER_DEFAULT = 'hyper_dflt'
# hyperparameters from stable-baselines zoo
HYPER_ZOO = 'hyper_zoo'
HYPER_PENG = 'hyper_dpmm'
HYPERS = HYPER_DEFAULT
# use_default_hypers = HYPERS == HYPER_DEFAULT

# number of training steps
mio_steps = {HYPER_DEFAULT:10, HYPER_PENG:6, HYPER_ZOO:2}[HYPERS]

algo = 'ppo2'
hyperparam = HYPERS

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

own_hypers = '' # f'zero_prct{s(zero_prct)}/' # f'slope{rem_slope}_init{s(rem_prct_init)}/' # f'scale{s(sftmx_scale)}_init{sftmx_init}_shft{s(sftmx_shift)}_invtmp{s(sftm_inv_temp)}/' # f'l1_lam{s(l1_scale)}/' # f'hid{s(hid_size)}/' # f'tpl{s(tuple_size)}/' #  f'sigm_sl{s(sigm_slope)}_th{s(sigm_thres)}/'
run = s(np.random.random_integers(0,1000))
info = 'pun100_nstp4096_gamma999'

# construct the paths
abs_project_path = dirname(dirname(dirname(__file__))) + '/'
save_path_norun= abs_project_path + \
                 f'models/monit/{approach}/{modification}/{env_name}/{n_parallel_envs}envs/{algo}/{hyperparam}/{mio_steps}mio/' \
                 + (f'{own_hypers + info}/' if len(own_hypers+info)>0 else '')
save_path = save_path_norun + f'{run}/'

# names of saved model before and after training
init_checkpoint = s(0)
final_checkpoint = s(999)

