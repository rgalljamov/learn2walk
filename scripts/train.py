import os.path
from scripts import eval
from scripts.common import config as cfg, utils
from scripts.common.schedules import LinearSchedule
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

    # create model directories
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
        os.makedirs(cfg.save_path + 'metrics')
        os.makedirs(cfg.save_path + 'models')
        os.makedirs(cfg.save_path + 'models/params')
        os.makedirs(cfg.save_path + 'envs')

    # setup environment
    env = utils.vec_env(cfg.env_id, norm_rew=True,
                        num_envs=cfg.n_envs)

    # setup model/algorithm
    training_timesteps = int(cfg.mio_steps * 1e6)
    learning_rate_schedule = LinearSchedule(cfg.lr_start*(1e-6), cfg.lr_final*(1e-6)).value
    network_args = {'net_arch': [{'vf': cfg.hid_layers, 'pi': cfg.hid_layers}],
                    'act_fun': tf.nn.relu}

    model = PPO2(MlpPolicy, env, verbose=1,
                 n_steps=int(cfg.batch_size/cfg.n_envs),
                 policy_kwargs=network_args,
                 learning_rate=learning_rate_schedule, ent_coef=cfg.ent_coef,
                 gamma=cfg.gamma, cliprange=cfg.cliprange,
                 tensorboard_log=cfg.save_path + 'tb_logs/')

    # automatically launch tensorboard
    run_tensorboard()

    # save model and weights before training
    utils.save_model(model, cfg.save_path, cfg.init_checkpoint)

    # train model
    model.learn(total_timesteps=training_timesteps, callback=TrainingMonitor())

    # save model after training
    utils.save_model(model, cfg.save_path, cfg.final_checkpoint)

    # close environment
    env.close()

    # evaluate last saved model
    eval.eval_model()
