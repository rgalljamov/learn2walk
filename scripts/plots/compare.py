# suppress the annoying TF Warnings at startup
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from scripts.plots.data_struct import Approach
from scripts.common.utils import config_pyplot
from scripts.plots import plot
import seaborn as sns
import numpy as np
plt = config_pyplot()

APD_BSLN = 'pd_bsln'
APD_NORM_ANGS = 'pd_norm_angs'
APD_NORM_ANGS_EXPMORE = 'pd_norm_angs_expmore'
APD_NORM_ANGS_SIGN_EXPMORE = 'pd_norm_angs_sign_expmore'
APD_NORM_DELTA = 'pd_norm_delta'
APT_BSLN = 'trq_bsln'

run_names_dict = {APD_BSLN: 'BSLN, init std = 1',
                  APD_NORM_ANGS: 'BSLN - normed target angles',
                  APD_NORM_ANGS_EXPMORE: 'BSLN - normed target angles - init std = 1',
                  APD_NORM_ANGS_SIGN_EXPMORE: 'BSLN - SIGN normed target angles - init std = 1',
                  APD_NORM_DELTA: 'normed deltas',
                  APT_BSLN: 'bsln'}

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

metrics = [MET_SUM_SCORE, MET_STABLE_WALKS, MET_STEP_REW, MET_MEAN_DIST, MET_REW_POS, MET_REW_VEL,
           MET_REW_COM, MET_TRAIN_EPRET, MET_TRAIN_STEPREW, MET_TRAIN_DIST, MET_TRAIN_EPLEN,
           MET_STEPS_TO_CONV, MET_PI_STD]

metric_names_dict = {MET_SUM_SCORE: 'Summary Score', MET_STABLE_WALKS: '# Stable Walks',
                MET_STEP_REW: 'Eval Step Reward', MET_MEAN_DIST: 'Eval Distance [m]',
                MET_REW_POS: 'Average Position Reward', MET_REW_VEL: 'Average Velocity Reward', 
                MET_REW_COM: 'Average COM Reward', MET_TRAIN_EPRET: 'Episode Return',
                MET_TRAIN_STEPREW: 'Train Step Reward', MET_TRAIN_DIST: 'Train Distance [m]',
                MET_TRAIN_EPLEN:'Train Episode Length', MET_STEPS_TO_CONV:'Steps to Convergence',
                MET_PI_STD: 'Policy STD'}


def check_data_for_completeness(approach, mio_train_steps=16):
    n_metrics = len(approach.metrics)
    for i, metric in enumerate(approach.metrics):
        subplot = plt.subplot(3, int(n_metrics / 3) + 1, i + 1)
        if isinstance(metric.mean, np.ndarray) and len(metric.mean) > 1:
            x = np.linspace(0, mio_train_steps*1e6, len(metric.mean))
            subplot.plot(x, metric.mean)
            subplot.plot(x, metric.mean_fltrd)
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, metric.mean_fltrd + metric.std, metric.mean_fltrd - metric.std,
                                 color=mean_color, alpha=0.25)
            tick_distance_mio = mio_train_steps / 4
        else:
            subplot.scatter(metric.data, np.arange(len(metric.data)))
        subplot.title.set_text(metric_names_dict[metric.label])
        subplot.set_xticks(np.arange(5) * tick_distance_mio * 1e6)
        subplot.set_xticklabels([f'{x}M' for x in np.arange(5) * tick_distance_mio])
    plt.show()


def download_approach_data(approach_name, project_name, run_name=None):
    train_steps = 16 if project_name == 'pd_approaches' else 8
    if run_name is None: run_name = run_names_dict[approach_name]
    approach = Approach(approach_name, project_name, run_name, metrics)
    check_data_for_completeness(approach, train_steps)
    approach.save()

def download_multiple_approaches(project_name, approach_names: list):
    run_names = [run_names_dict[approach] for approach in approach_names]
    for i in range(len(run_names)):
        approach_name = approach_names[i]
        print('Downloading approach:', approach_name)
        download_approach_data(approach_name, project_name, run_names[i])

def download_PD_approaches():
    project_name = "pd_approaches"
    ap_names = [APD_BSLN, APD_NORM_ANGS, APD_NORM_ANGS_EXPMORE,
                APD_NORM_ANGS_SIGN_EXPMORE, APD_NORM_DELTA]
    download_multiple_approaches(project_name, ap_names)

def compare_PD_approaches():
    ap_names = [APD_BSLN, APD_NORM_ANGS, APD_NORM_DELTA]
    aps = [Approach(name) for name in ap_names]
    metric_labels = [MET_TRAIN_EPRET]
    n_metrics = len(metric_labels)
    n_steps = 16e6
    for i, metric_label in enumerate(metric_labels):
        subplot = plt.subplot(1, n_metrics, i + 1)
        metrics = [metric for ap in aps for metric in ap.metrics
                   if metric.label == metric_label]
        for metric in metrics:
            x = np.linspace(0, n_steps, len(metric.mean))
            mean = metric.mean_fltrd
            subplot.plot(x, mean)
            std = metric.std_fltrd
            mean_color = subplot.get_lines()[-1].get_color()
            subplot.fill_between(x, mean + std, mean - std,
                                 color=mean_color, alpha=0.25)
            # subplot.plot(x, metric.mean, color=mean_color, alpha=0.4)
            subplot.set_xticks(np.arange(5) * 4e6)
            subplot.set_xticklabels([f'{x}M' for x in np.arange(5) * 4])
            # subplot.set_yticks((np.arange(3)+1)*0.25)
            subplot.set_title(metric_names_dict[metric.label])
            subplot.legend(ap_names)
    plt.show()


if __name__ == '__main__':
    # compare_PD_approaches()
    # download_PD_approaches()
    # download_approach_data(AP_NORM_ANGS, 'pd_approaches', run_names_dict[AP_NORM_ANGS])
    download_approach_data(APT_BSLN, 'final3d_trq')
