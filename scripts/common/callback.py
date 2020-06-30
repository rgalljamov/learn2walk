from os import makedirs
import tensorflow as tf
import numpy as np

from scripts.common import config as cfg, utils
from stable_baselines.common.callbacks import BaseCallback

class TrainingMonitor(BaseCallback):
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TrainingMonitor, self).__init__(verbose)

    def _on_training_start(self) -> None:
        self.env = self.training_env

    def _on_step(self) -> bool:
        ep_len = np.mean(self.env.get_attr('ep_len_smoothed'))
        ep_ret = np.mean(self.env.get_attr('ep_ret_smoothed'))

        # Log scalar value (here a random variable)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='own_data/mean_smoothed_episode_length', simple_value=ep_len),
            tf.Summary.Value(tag='own_data/mean_smoothed_episode_return', simple_value=ep_ret)])

        self.locals['writer'].add_summary(summary, self.num_timesteps)

        return True


def _save_rews_n_rets(locals):
    # save all rewards and returns of the training, batch wise
    path_rews = cfg.save_path + 'metrics/train_rews.npy'
    path_rets = cfg.save_path + 'metrics/train_rets.npy'

    try:
        # load already saved rews and rets
        rews = np.load(path_rews)
        rets = np.load(path_rets)
        # combine saved with new rews and rets
        rews = np.concatenate((rews, locals['true_reward']))
        rets = np.concatenate((rets, locals['returns']))
    except Exception:
        rews = locals['true_reward']
        rets = locals['returns']

    # save
    np.save(path_rets, np.float16(rets))
    np.save(path_rews, np.float16(rews))


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    Used to log relevant information during training
    :param _locals: (dict)
    :param _globals: (dict)
    """

    # save all rewards and returns of the training, batch wise
    _save_rews_n_rets(_locals)

    # Log other data about every 200k steps
    # todo: calc as a function of batch for ppo
    #  when updating stable-baselines doesn't provide another option
    #  and check how often TD3 and SAC raise the callback.
    saving_interval = 390 if cfg.use_default_hypers else 6
    n_updates = _locals['update']
    if n_updates % saving_interval == 0:

        model = _locals['self']
        utils.save_pi_weights(model, n_updates)

        # save the model and environment only for every second update (every 400k steps)
        if n_updates % (2*saving_interval) == 0:
            # save model
            model.save(save_path=cfg.save_path + 'models/model_' + str(n_updates))
            # save env
            env_path = cfg.save_path + 'envs/' + 'env_' + str(n_updates)
            makedirs(env_path)
            # save Running mean of observations and reward
            env = model.get_env()
            env.save_running_average(env_path)
            utils.log("Saved model after {} updates".format(n_updates))

    return True