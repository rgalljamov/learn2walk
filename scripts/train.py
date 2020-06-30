import os.path
from scripts import eval
from scripts.common import config as cfg, utils
from scripts.common.callback import TrainingMonitor

# to decrease the amount of deprecation warnings
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.logging.set_verbosity(tf.logging.ERROR)

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy


def run_tensorboard():
    import os, threading
    print('You can start tensorboard with the following command:\n'
          'tensorboard --logdir="' + cfg.save_path + 'tb_logs/"')
    tb_path = '/home/rustam/anaconda3/envs/drl/bin/tensorboard ' if utils.is_remote() \
        else '/home/rustam/.conda/envs/tensorflow/bin/tensorboard '
    tb_thread = threading.Thread(
        target=lambda: os.system(tb_path + '--logdir="' + cfg.save_path + 'tb_logs/"'),
        daemon=True)
    tb_thread.start()


if __name__ == "__main__":

    # create direction for the model
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
        os.makedirs(cfg.save_path + 'metrics')
        os.makedirs(cfg.save_path + 'models')
        os.makedirs(cfg.save_path + 'models/params')
        os.makedirs(cfg.save_path + 'envs')

    env = utils.vec_env(cfg.env_id, num_envs=cfg.n_parallel_envs, norm_rew=True)

    if cfg.hyperparam == cfg.HYPER_DEFAULT:
        utils.log('Training with default params from Stable Baselines')
        model = PPO2(MlpPolicy, env, verbose=1, n_steps=4096, gamma=0.999,
                     tensorboard_log=cfg.save_path + 'tb_logs/')
    elif cfg.hyperparam == cfg.HYPER_PENG:
        model = PPO2(MlpPolicy, env,
                     n_steps=4096, nminibatches=16, lam=0.95, verbose=1,
                     gamma=0.95, learning_rate=5e-5, cliprange=0.2,
                     tensorboard_log=cfg.save_path + 'tb_logs/')
    elif cfg.hyperparam == cfg.HYPER_ZOO:
        model = PPO2(MlpPolicy, env,
                     n_steps=1024, nminibatches=64, lam=0.95, verbose=1,
                     gamma=0.90, learning_rate=0.00025, cliprange=0.1,
                     noptepochs=10, ent_coef=0, cliprange_vf=-1,
                     tensorboard_log=cfg.save_path + 'tb_logs/')

    # automatically launch tensorboard
    run_tensorboard()

    # save model and weights before training
    utils.save_model(model, cfg.save_path, cfg.init_checkpoint)

    # train model
    model.learn(total_timesteps=int(cfg.mio_steps * 1e6), callback=TrainingMonitor())

    # save model after training
    utils.save_model(model, cfg.save_path, cfg.final_checkpoint)

    # close envioronment
    env.close()

    # evaluate last saved model
    # todo: evaluate multiple models, if previous models were better
    eval.eval_model()
