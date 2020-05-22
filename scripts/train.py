import os.path
import scripts.common.config as cfg
from scripts.common import utils
from scripts.common.callback import callback
from scripts import eval

# to decrease the amount of deprecation warnings
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.logging.set_verbosity(tf.logging.ERROR)

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy


def run_tensorboard():
    import os, threading
    tb_thread = threading.Thread(
        target=lambda: os.system('/home/rustam/anaconda3/envs/drl/bin/tensorboard '
                                 '--logdir=' + cfg.save_path + "tb_logs/"),
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

    if cfg.use_default_hypers:
        utils.log('Training with default params from Stable Baselines')
        model = PPO2(MlpPolicy, env, verbose=1, n_steps=512,
                     tensorboard_log=cfg.save_path + 'tb_logs/')
    else:
        model = PPO2(MlpPolicy, env,
                     n_steps=8192, nminibatches=32, lam=0.95, verbose=1,
                     gamma=0.99, noptepochs=10, ent_coef=0.001, learning_rate=2.5e-4, cliprange=0.2,
                     tensorboard_log=cfg.save_path + 'tb_logs/')

    # automatically launch tensorboard
    run_tensorboard()

    # save model and weights before training
    utils.save_model(model, cfg.save_path, cfg.init_checkpoint)

    # train model
    model.learn(total_timesteps=int(cfg.mio_steps * 1e6)) #, callback=callback)

    # save model after training
    utils.save_model(model, cfg.save_path, cfg.final_checkpoint)

    # evaluate last saved model
    # todo: evaluate multiple models, if previous models were better
    eval.eval_model()
