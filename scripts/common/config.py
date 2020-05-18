import numpy as np

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

# choose approach
AP_ORIGINAL = 'orig'
approach = 'test'

# config environment
n_parallel_envs = 4
envs = ['Walker2d-v3', 'Humanoid-v3', 'Blind-BipedalWalker-v2', 'BipedalWalker-v2']
env_names = ['walker2d', 'humanoid', 'blind_walker', 'walker']
env_index = 0
env_id = envs[env_index]
env_name = env_names[env_index]

# default hyperparameters from stable-baselines
HYPER_DEFAULT = 'hyper_default'
# hyperparameters from stable-baselines zoo
HYPER_ZOO = 'hyper_zoo'
HYPERS = HYPER_DEFAULT
use_default_hypers = HYPERS == HYPER_DEFAULT

# number of training steps
mio_steps = 0.5 if use_default_hypers else 6

algo = 'ppo2'
hyperparam = 'zoo' if not use_default_hypers else 'default'

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
info = ''
save_path_norun= f'../models/{approach}/{env_name}/{algo}/{hyperparam}/{mio_steps}mio/' \
                 + (f'{own_hypers + info}/' if len(own_hypers+info)>0 else '')
save_path = save_path_norun + f'{run}/'

# names of saved model before and after training
init_checkpoint = s(0)
final_checkpoint = s(999)

