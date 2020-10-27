# suppress the annoying TF Warnings at startup
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from scripts.plots.data_struct import Approach
from scripts.plots.wandb_api import Api
from scripts.common.utils import config_pyplot
import numpy as np
plt = config_pyplot()

PR_PD_APS = "pd_approaches"

# metric labels
MET_SUM_SCORE = '_det_eval/1. AUC stable walks count'
MET_STABLE_WALKS = '_det_eval/2. stable walks count'
MET_STEP_REW = '_det_eval/3. mean step reward'
MET_MEAN_DIST = '_det_eval/4. mean eval distance'
MET_REW_POS = '_rews/1. mean ep pos rew (8envs, smoothed 0.9)'
MET_REW_VEL = '_rews/2. mean ep vel rew (8envs, smoothed 0.9)'
MET_REW_COM = '_rews/3. mean ep com rew (8envs, smoothed 0.9)'
MET_TRAIN_EPRET = '_train/4. episode return (smoothed 0.75)'
MET_TRAIN_STEPREW = '_train/3. step reward (smoothed 0.25)'
MET_TRAIN_DIST = '_train/1. moved distance (stochastic, smoothed 0.25)'
MET_TRAIN_EPLEN = '_train/2. episode length (smoothed 0.75)'
MET_STEPS_TO_CONV = 'log_steps_to_convergence'
MET_PI_STD = 'acts/1. mean std of 8 action distributions'

metric_names = {MET_SUM_SCORE: 'Summary Score', MET_STABLE_WALKS: '# Stable Walks',
                MET_STEP_REW: 'Eval Step Reward', MET_MEAN_DIST: 'Eval Distance [m]',
                MET_REW_POS: 'Average Position Reward', MET_REW_VEL: 'Average Velocity Reward', 
                MET_REW_COM: 'Average COM Reward', MET_TRAIN_EPRET: 'Episode Return',
                MET_TRAIN_STEPREW: 'Train Step Reward', MET_TRAIN_DIST: 'Train Distance [m]',
                MET_TRAIN_EPLEN:'Train Episode Length', MET_STEPS_TO_CONV:'Steps to Convergence',
                MET_PI_STD: 'Policy STD'}

project_name = PR_PD_APS
run_name = 'normed deltas' # 'BSLN - normed target angles', 'normed deltas'
metrics = [MET_SUM_SCORE, MET_STABLE_WALKS, MET_STEP_REW, MET_MEAN_DIST, MET_REW_POS, MET_REW_VEL,
           MET_REW_COM, MET_TRAIN_EPRET, MET_TRAIN_STEPREW, MET_TRAIN_DIST, MET_TRAIN_EPLEN,
           MET_STEPS_TO_CONV, MET_PI_STD]

if __name__ == '__main__':
    api = Api(project_name)
    approach = Approach('PD_NORM_DELTA', PR_PD_APS, run_name, metrics)
    # approach.save()
    # exit(33)
    n_metrics = len(approach.metrics)
    for i, metric in enumerate(approach.metrics):
        subplot = plt.subplot(3, int(n_metrics/3)+1, i+1)
        subplot.plot(np.linspace(0, 16e6, len(metric.data[0,:])), metric.data[0,:])
        subplot.title.set_text(metric_names[metric.label])
    plt.show()
    exit(33)

# stable_walks = api.get_metrics(run_names, metric_name)
#
# print('shape stable walks is ', stable_walks.shape)
# for i in range(stable_walks.shape[0]):
#     plt.plot(stable_walks[i,:])
# mean = np.mean(stable_walks, axis=0)
# std = np.std(stable_walks, axis=0)
# print('mean shape is ', mean.shape)
# plt.plot(mean)
# plt.show()

