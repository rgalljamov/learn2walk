# suppress the annoying TF Warnings at startup
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from scripts.plots.data_struct import Approach
from scripts.plots.wandb_api import Api
from scripts.common.utils import config_pyplot
from scripts.plots import plot
import seaborn as sns
import numpy as np
plt = config_pyplot()
sns.set_style("whitegrid", {'axes.edgecolor': '#ffffff00'})

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

def check_data_for_completeness(approach):
    n_metrics = len(approach.metrics)
    for i, metric in enumerate(approach.metrics):
        subplot = plt.subplot(3, int(n_metrics / 3) + 1, i + 1)
        if isinstance(metric.mean, np.ndarray) and len(metric.mean) > 1:
            x = np.linspace(0, 16e6, len(metric.mean))
            subplot.plot(x, metric.mean)
            subplot.plot(x, metric.mean_fltrd)
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, metric.mean_fltrd + metric.std, metric.mean_fltrd - metric.std,
                                 color=mean_color, alpha=0.25)
        else:
            subplot.scatter(metric.data, np.arange(len(metric.data)))
        subplot.title.set_text(metric_names[metric.label])
    plt.show()


def download_approach_data():
    AP_NAME = 'PD_BSLN'
    project_name = "pd_approaches"
    run_name = 'BSLN, init std = 1'  # 'BSLN - normed target angles', 'normed deltas'
    metrics = [MET_SUM_SCORE, MET_STABLE_WALKS, MET_STEP_REW, MET_MEAN_DIST, MET_REW_POS, MET_REW_VEL,
               MET_REW_COM, MET_TRAIN_EPRET, MET_TRAIN_STEPREW, MET_TRAIN_DIST, MET_TRAIN_EPLEN,
               MET_STEPS_TO_CONV, MET_PI_STD]
    approach = Approach(AP_NAME, project_name, run_name, metrics)
    check_data_for_completeness(approach)
    approach.save()

if __name__ == '__main__':
    download_approach_data()
