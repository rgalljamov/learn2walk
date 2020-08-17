from scripts.behavior_cloning.dataset import get_refs, get_refs_stats
from scripts.common.utils import vec_env, log
from scripts.common import config as cfg

from matplotlib import pyplot as plt
from os.path import dirname
import numpy as np

OVERWRITE_OBS_RMS = False
DEBUG = False

def get_obs_rms(do_log=False):
    """:returns (mean, var) of pretrained obs_rms. Path defined in cfg.
    Pretrained statistics might come from refs or from previous runs."""
    model_path = cfg.init_obs_rms_path
    env = vec_env(cfg.env_id, norm_rew=True, num_envs=cfg.n_envs,
                  deltas=cfg.is_mod(cfg.MOD_PI_OUT_DELTAS), load_path=model_path)

    if do_log: log('Successfully loaded pretrained OBS_RMS:',
                   [f'object:\t {env.obs_rms}',
                    f'mean:\t {env.obs_rms.mean}',
                    f'var:\t {env.obs_rms.var}'])

    return env.obs_rms.mean, env.obs_rms.var


# TODO: NOT USING OBS_STDS FROM REFERENCE TRAJECTORIES BUT FROM A PREVIOUS RUN.
if __name__ == '__main__':
    # build VecNormalize wrapped Environment
    env = vec_env(cfg.env_id, norm_rew=True, num_envs=cfg.n_envs,
                            deltas=cfg.is_mod(cfg.MOD_PI_OUT_DELTAS))
    # get refs statistics
    ref_means, ref_vars, ref_stds = get_refs_stats(None, False, False)
    # add phase variable statistics
    phase = np.linspace(0, 1, 250)
    phase_mean, phase_var = np.mean(phase), np.var(phase)
    # add desired walking speed statistics
    walk_speeds = get_refs()._calculate_walking_speed()
    speed_mean, speed_var = np.mean(walk_speeds), np.var(walk_speeds)
    # concatenate means and vars
    ref_means = np.concatenate([np.array([phase_mean, speed_mean]), ref_means])
    ref_vars = np.concatenate([np.array([phase_var, speed_var]), ref_vars])

    if DEBUG:
        plt.subplot(1,2,1)
        plt.title('Phase Variable Statistics')
        plt.plot(phase)
        plt.plot(range(250), np.ones((250,))*phase_mean)
        plt.fill_between(range(250), phase_mean - np.sqrt(phase_var),
                         phase_mean + np.sqrt(phase_var), alpha=0.25)
        plt.subplot(1,2,2)
        plt.title('Walking Speed Statistics')
        plt.plot(walk_speeds)
        x_len = len(walk_speeds)
        plt.plot(range(x_len), np.ones((x_len,)) * speed_mean)
        plt.fill_between(range(x_len), speed_mean - np.sqrt(speed_var),
                         speed_mean + np.sqrt(speed_var), alpha=0.25)
        plt.show()

    # compare
    labels = get_refs().get_kinematics_labels()
    refs_mean = ref_means
    refs_var = ref_vars
    run_mean = env.obs_rms.mean
    run_var = env.obs_rms.var

    deltas_mean = np.abs(run_mean - ref_means)
    deltas_var = np.abs(run_var - ref_vars)

    deltas_mean_prct = np.array((100 * deltas_mean/np.abs(refs_mean)), dtype=int)
    deltas_var_prct = np.array(100 * deltas_var/np.abs(run_var), dtype=int)

    for i, label in enumerate(labels):
        print('\n', label)
        print('delta_prct \t delta_mean \t run_mean \t ref_mean')
        print('%d \t\t\t %.3f \t\t\t %.3f \t\t\t %.3f' % (deltas_mean_prct[i], deltas_mean[i], run_mean[i], refs_mean[i]))
    if not OVERWRITE_OBS_RMS: SystemExit('Expectedly finished script W/O saving new refs!')

    env.obs_rms.mean = ref_means
    env.obs_rms.var = ref_vars

    # construct save paths
    model_path = dirname(dirname(dirname(__file__))) + '/scripts/behavior_cloning/models/rms/'
    model_name = 'ref_obs_rms_init_const_speed'
    save_path = model_path + model_name
    env.save(save_path)
