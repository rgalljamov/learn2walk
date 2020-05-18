import gym, os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.common.config import save_path
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def config_plots():
    """ set desired plotting settings """

    PLOT_FONT_SIZE = 24
    PLOT_TICKS_SIZE = 18
    PLOT_LINE_WIDTH = 2

    # activate and configure seaborn style for plots
    sns.set()
    sns.set_context(rc={"lines.linewidth": PLOT_LINE_WIDTH, 'xtick.labelsize': PLOT_TICKS_SIZE,
                        'ytick.labelsize': PLOT_TICKS_SIZE, 'savefig.dpi': 1024,
                        'axes.titlesize': PLOT_TICKS_SIZE, 'figure.autolayout': True,
                        'legend.fontsize': PLOT_FONT_SIZE - 4, 'axes.labelsize': PLOT_FONT_SIZE})

    # configure saving format and directory
    PLOT_FIGURE_SAVE_FORMAT = 'png'  # 'pdf' #'eps'
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'savefig.format': PLOT_FIGURE_SAVE_FORMAT})
    plt.rcParams.update({"savefig.directory": '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/figures'})


def vec_env(env_name, num_envs=4, seed=33, norm_rew=True):
    '''creates environments, vectorizes them and sets different seeds
    @:param norm_rew: reward should only be normalized during training'''

    def make_env_func(env_name, seed, rank):
        def make_env():
            env = gym.make(env_name)
            env.seed(seed+rank*100)
            return env
        return make_env

    if num_envs == 1:
        vec_env = DummyVecEnv([make_env_func(env_name, seed, 0)])
    else:
        env_fncts = [make_env_func(env_name, seed, rank) for rank in range(num_envs)]
        vec_env = SubprocVecEnv(env_fncts)

    # normalize environments
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=norm_rew)

    return vec_env


def log(text):
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
    model.save(save_path=path+'models/model_' + str(checkpoint))
    save_pi_weights(model, checkpoint)
    # save Running mean of observations and reward
    env_path = save_path + 'envs/env_' + str(checkpoint)
    os.makedirs(env_path)
    model.get_env().save_running_average(env_path)


def save_pi_weights(model, name):
    """Saves all weights of the policy network
     @:param name: Info to append to the file's name"""
    weights = []
    biases = []
    attens = []
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