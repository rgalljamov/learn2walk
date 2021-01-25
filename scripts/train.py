import os.path
import wandb
from scripts import eval
from scripts.common import config as cfg, utils
from scripts.common.schedules import LinearDecay, ExponentialSchedule
from scripts.common.callback import TrainingMonitor
from scripts.common.policies import CustomPolicy
from scripts.common.distributions import LOG_STD_MIN, LOG_STD_MAX

# to decrease the amount of deprecation warnings
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.logging.set_verbosity(tf.logging.ERROR)

from scripts.algos.custom_ppo2 import CustomPPO2
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


def init_wandb(model):
    batch_size = model.n_steps * model.n_envs
    params = {
        "path": cfg.save_path,
        "mod": cfg.modification,
        "ctrl_freq": cfg.CTRL_FREQ,
        "lr0": cfg.lr_start,
        "lr1": cfg.lr_final,
        'hid_sizes': cfg.hid_layer_sizes_vf,
        'hid_sizes_vf': cfg.hid_layer_sizes_vf,
        'hid_sizes_pi': cfg.hid_layer_sizes_pi,
        'peak_joint_torques': cfg.peak_joint_torques,
        "noptepochs": cfg.noptepochs,
        "batch_size": batch_size,
        "cfg.batch_size": cfg.batch_size,
        "n_mini_batches": model.nminibatches,
        "cfg.minibatch_size": cfg.minibatch_size,
        "mini_batch_size": int(batch_size / model.nminibatches),
        "mio_steps": cfg.mio_samples,
        "ent_coef": model.ent_coef,
        "ep_dur": cfg.ep_dur_max,
        "imit_rew": cfg.rew_weights,
        "logstd": cfg.init_logstd,
        "min_logstd": LOG_STD_MIN,
        "max_logstd": LOG_STD_MAX,
        "env": cfg.env_abbrev,
        "gam": model.gamma,
        "lam": model.lam,
        "n_envs": model.n_envs,
        "seed": model.seed,
        "policy": model.policy,
        "n_steps": model.n_steps,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        "nminibatches": model.nminibatches,
        "clip0": cfg.clip_start,
        "clip1": cfg.clip_end,
        "n_cpu_tf_sess": model.n_cpu_tf_sess}

    if cfg.is_mod(cfg.MOD_REFS_RAMP):
        params['skip_n_steps'] = cfg.SKIP_N_STEPS
        params['steps_per_vel'] = cfg.STEPS_PER_VEL

    if cfg.is_mod(cfg.MOD_E2E_ENC_OBS):
        params['enc_layers'] = cfg.enc_layer_sizes

    wandb.init(config=params, sync_tensorboard=True, name=cfg.get_wb_run_name(),
               project=cfg.wb_project_name, notes=cfg.wb_run_notes)


def train():

    # create model directories
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
        os.makedirs(cfg.save_path + 'metrics')
        os.makedirs(cfg.save_path + 'models')
        os.makedirs(cfg.save_path + 'models/params')
        os.makedirs(cfg.save_path + 'envs')

    # setup environment
    env = utils.vec_env(cfg.env_id, norm_rew=True, num_envs=cfg.n_envs)

    # setup model/algorithm
    training_timesteps = int(cfg.mio_samples * 1e6)
    lr_start = cfg.lr_start
    lr_end = cfg.lr_final

    learning_rate_schedule = LinearDecay(lr_start, lr_end).value
    clip_schedule = ExponentialSchedule(cfg.clip_start, cfg.clip_end, cfg.clip_exp_slope).value

    network_args = {'net_arch': [{'vf': cfg.hid_layer_sizes_vf, 'pi': cfg.hid_layer_sizes_pi}],
                    'act_fun': tf.nn.relu} if not cfg.is_mod(cfg.MOD_CUSTOM_POLICY) else {}

    model = CustomPPO2(CustomPolicy if cfg.is_mod(cfg.MOD_CUSTOM_POLICY) else MlpPolicy,
                       env, verbose=1, n_steps=int(cfg.batch_size/cfg.n_envs),
                       policy_kwargs=network_args,
                       learning_rate=learning_rate_schedule, ent_coef=cfg.ent_coef,
                       gamma=cfg.gamma, noptepochs=cfg.noptepochs,
                       cliprange_vf=clip_schedule if cfg.is_mod(cfg.MOD_CLIPRANGE_SCHED) else cfg.cliprange,
                       cliprange=clip_schedule if cfg.is_mod(cfg.MOD_CLIPRANGE_SCHED) else cfg.cliprange,
                       tensorboard_log=cfg.save_path + 'tb_logs/')

    # init wandb
    if not cfg.DEBUG: init_wandb(model)

    # automatically launch tensorboard, only if wandb is not used!
    # otherwise wandb automatically uploads all TB logs to wandb
    # run_tensorboard()

    # save model and weights before training
    if not cfg.DEBUG:
        utils.save_model(model, cfg.save_path, cfg.init_checkpoint)

    # train model
    model.learn(total_timesteps=training_timesteps, callback=TrainingMonitor())

    # save model after training
    utils.save_model(model, cfg.save_path, cfg.final_checkpoint)

    # close environment
    env.close()

    # evaluate last saved model
    eval.eval_model()


if __name__ == '__main__':
    train()