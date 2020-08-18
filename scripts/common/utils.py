import gym, os, wandb
import numpy as np
import seaborn as sns
from os import path, getcwd

def is_remote():
    # automatically detect running PC
    return 'remote' in path.abspath(getcwd())

def get_absolute_project_path():
    dirname = path.dirname
    return dirname(dirname(dirname(__file__))) + '/'

abs_project_path = get_absolute_project_path()
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
# import gym_mimic_envs

# used for running mean
_running_means = {}

# used for exponential running smoothing
_exp_weighted_averages = {}

def import_pyplot():
    """Imports pyplot and activates the right backend
       to render plots on local system even they're drawn remotely.
       @param: setup_plot_params: if true, activates seaborn and sets rcParams"""
    import matplotlib
    try:
        matplotlib.use('tkagg') if is_remote() else matplotlib.use('Qt5Agg')
    except Exception:
        pass
    from matplotlib import pyplot as plt
    return plt

plt = import_pyplot()

PLOT_FONT_SIZE = 24
PLOT_TICKS_SIZE = 18
PLOT_LINE_WIDTH = 2

def config_pyplot(fullscreen=False, font_size=PLOT_TICKS_SIZE, tick_size=PLOT_TICKS_SIZE,
                  legend_fontsize=PLOT_FONT_SIZE-4):
    """ set desired plotting settings and returns a pyplot object
     @ return: pyplot object with seaborn style and configured rcParams"""

    # activate and configure seaborn style for plots
    sns.set()
    sns.set_context(rc={"lines.linewidth": PLOT_LINE_WIDTH, 'xtick.labelsize': tick_size,
                        'ytick.labelsize': tick_size, 'savefig.dpi': 1024,
                        'axes.titlesize': tick_size, 'figure.autolayout': True,
                        'legend.fontsize': legend_fontsize, 'axes.labelsize': font_size})

    # configure saving format and directory
    PLOT_FIGURE_SAVE_FORMAT = 'png'  # 'pdf' #'eps'
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'savefig.format': PLOT_FIGURE_SAVE_FORMAT})
    plt.rcParams.update({"savefig.directory": '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/figures'})

    if fullscreen:
        # plot figure in full screen mode (scaled down aspect ratio of my screen)
        plt.rcParams['figure.figsize'] = (19.2, 10.8)

    return plt

def vec_env(env_name, num_envs=4, seed=33, norm_rew=True,
            load_path=None, deltas=False):
    '''creates environments, vectorizes them and sets different seeds
    :param norm_rew: reward should only be normalized during training
    :param load_path: if set, the VecNormalize environment will
                      load the running means from this path.
    :param deltas: if True, set the action space of the environment
                   to max joint deltas inctead of joint angles
    :returns: VecNormalize (wrapped Subproc- or Dummy-VecEnv) '''

    from gym_mimic_envs.mimic_env import MimicEnv
    from gym_mimic_envs.monitor import Monitor as EnvMonitor

    def make_env_func(env_name, seed, rank):
        def make_env():
            env = gym.make(env_name)
            if isinstance(env, MimicEnv):
                # wrap a MimicEnv in the EnvMonitor
                # has to be done before converting into a VecEnv!
                env = EnvMonitor(env)
                if deltas:
                    env.set_action_space_deltas()
            env.seed(seed+rank*100)
            return env
        return make_env

    if num_envs == 1:
        vec_env = DummyVecEnv([make_env_func(env_name, seed, 0)])
    else:
        env_fncts = [make_env_func(env_name, seed, rank) for rank in range(num_envs)]
        vec_env = SubprocVecEnv(env_fncts)

    # normalize environments
    if load_path is not None:
        vec_normed = VecNormalize.load(load_path, vec_env)
    else:
        try:
            from scripts.common.config import is_mod, MOD_LOAD_OBS_RMS
            if not is_mod(MOD_LOAD_OBS_RMS): raise Exception
            # load the obs_rms from a previously trained model
            init_obs_rms_path = abs_project_path + \
                                'scripts/behavior_cloning/models/rms/env_999'
            vec_normed = VecNormalize.load(init_obs_rms_path, vec_env)
            log('Successfully loaded OBS_RMS from a previous model:',
                [f'file:\t {init_obs_rms_path}',
                 f'mean:\t {vec_normed.obs_rms.mean}',
                 f'var:\t {vec_normed.obs_rms.var}'])
        except:
            log('Do NOT loading obs_rms from a previous run.')
            vec_normed = VecNormalize(vec_env, norm_obs=True, norm_reward=norm_rew)

    return vec_normed


def check_environment(env_name):
    from gym_mimic_envs.monitor import Monitor as EnvMonitor
    from stable_baselines.common.env_checker import check_env
    env = gym.make(env_name)
    log('Checking custom environment')
    check_env(env)
    env = EnvMonitor(env)
    log('Checking custom env in custom monitor wrapper')
    check_env(env)
    exit(33)


def log(text, input_list=None):

    if input_list is not None and isinstance(input_list, list):
        list_as_str = '\n'.join([str(item) for item in input_list])
        text += '\n' + list_as_str

    print("\n---------------------------------------\n"
          + text +
          "\n---------------------------------------\n")


def plot_weight_matrix(weight_matrix, show=True, max_abs_value=1, center_cmap=True):
    ''':param center_cmap: use a colormap where zero should correspond to white
       :param max_abs_value: min and max possible value on the centered cmap
       :param show: show plot or not - set to false when called in a loop
       :returns the passed weight matrix (saves one line during plotting)
    '''
    if center_cmap:
        plt.pcolor(weight_matrix, vmin=-max_abs_value, vmax=max_abs_value, cmap='seismic')
    else:
        plt.pcolor(weight_matrix, cmap='jet', ec='#eeeeee')
    plt.colorbar()
    if show: plt.show()
    return weight_matrix


def save_model(model, path, checkpoint):
    """ saves the model, the corresponding environment means and pi weights"""
    model_path = path + f'models/model_{checkpoint}'
    model.save(save_path=model_path)
    save_pi_weights(model, checkpoint)
    # save Running mean of observations and reward
    env_path = path + f'envs/env_{checkpoint}'
    model.get_env().save(env_path)
    # save model and env to wandb
    wandb.save(model_path+'.zip')
    wandb.save(env_path)


def save_pi_weights(model, name):
    """Saves all weights of the policy network
     @:param name: Info to append to the file's name"""
    weights = []
    biases = []
    attens = []

    # todo: check why it does not work for pretrained models!
    return

    save_path = None
    # log('Model Parameters:', model.params)

    for param in model.params:
        if 'pi' in param.name:
            if 'w:0' in param.name:
                weights.append(model.sess.run(param))
            elif 'b:0' in param.name:
                biases.append(model.sess.run(param))
            elif 'att' in param.name:
                print('Saving attention matrix!')
                attens.append(model.sess.run(param))

    if len(weights) > 10:
        # we have a sparse network
        np.savez(save_path + 'models/params/weights_' + str(name), Ws=weights)
        np.savez(save_path + 'models/params/biases_' + str(name), bs=biases)
        print('Saved weights of a sparse network')
        return

    # save policy network weights
    np.savez(save_path + 'models/params/weights_' + str(name),
             W0=weights[0], W1=weights[1], W2=weights[2])
    np.savez(save_path + 'models/params/biases_' + str(name),
             b0=biases[0], b1=biases[1], b2=biases[2])
    if len(attens) > 1:
        np.savez(save_path + 'models/params/attens_' + str(name),
                 A0=attens[0], A1=attens[1])

def load_env(checkpoint, save_path, env_id):
    # load a single environment for evaluation
    env_path = save_path + f'envs/env_{checkpoint}'
    env = vec_env(env_id, num_envs=1, norm_rew=False, load_path=env_path)
    # set the calculated running means for obs and rets
    # env.load(env_path)
    return env

def smooth_exponential(data, alpha=0.9):
    smoothed = np.copy(data)
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1-alpha) * smoothed[t-1]
    return smoothed

def lowpass_filter_data(data, sample_rate, cutoff_freq, order=1):
    """
    Uses a butterworth filter to filter data in both directions without causing any delay.
    """
    from scipy import signal

    nyquist_freq = sample_rate/2
    norm_cutoff_freq = cutoff_freq/nyquist_freq

    b, a = signal.butter(order, norm_cutoff_freq, 'low')
    fltrd_data = signal.filtfilt(b, a, data)

    return fltrd_data


def running_mean(label: str, new_value):
    """
    Computes the running mean, given a new value.
    Several running means can be monitored parallely by providing different labels.
    :param label: give your running mean a name.
                  Will be used as a dict key to save current running mean value.
    :return: current running mean value for the provided label
    """

    if label in  _running_means:
        old_mean, num_values = _running_means[label]
        new_mean = (old_mean * num_values + new_value) / (num_values + 1)
        _running_means[label] = [new_mean, num_values + 1]
    else:
        # first value for the running mean
        _running_means[label] = [new_value, 1]

    return _running_means[label][0]


def exponential_running_smoothing(label, new_value, smoothing_factor=0.9):
    """
    Implements an exponential running smoothing filter.
    Several inputs can be filtered parallely by providing different labels.
    :param label: give your filtered data a name.
                  Will be used as a dict key to save current filtered value.
    :return: current filtered value for the provided label
    """
    global _exp_weighted_averages

    if label not in _exp_weighted_averages:
        _exp_weighted_averages[label] = new_value
        return new_value

    new_average = smoothing_factor * new_value + (1 - smoothing_factor) * _exp_weighted_averages[label]
    _exp_weighted_averages[label] = new_average

    return new_average


def resetExponentialRunningSmoothing(label):
    """
    Sets the current value of the exponential running smoothing identified by the label to zero.
    """
    global _exp_weighted_averages
    _exp_weighted_averages[label] = 0
    return True