import os.path
import wandb
from scripts import eval
from scripts.common import config as cfg, utils
from scripts.common.schedules import LinearSchedule
from scripts.common.callback import TrainingMonitor
from scripts.common.policies import CustomPolicy

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
        "lr0": cfg.lr_start,
        "lr1": cfg.lr_final,
        'hid_sizes': cfg.hid_layer_sizes,
        "noptepochs": cfg.noptepochs,
        "batch_size": batch_size,
        "n_mini_batches": model.nminibatches,
        "mini_batch_size": int(batch_size / model.nminibatches),
        "mio_steps": cfg.mio_steps,
        "mio_steps_to_lr1": cfg.mio_steps_to_lr1,
        "lr_slope": cfg.slope,
        "ent_coef": model.ent_coef,
        "ep_dur": cfg.ep_dur_max,
        "imit_rew": cfg.rew_weights,
        "logstd": cfg.logstd,
        "et_rew": cfg.et_reward,
        "ep_end_rew": cfg.ep_end_reward,
        "et_rew_thres": cfg.et_rew_thres,
        "env": cfg.env_name,
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
        "cliprange_vf": model.cliprange_vf,
        "n_cpu_tf_sess": model.n_cpu_tf_sess}

    if cfg.is_mod(cfg.MOD_REFS_RAMP):
        params['skip_n_steps'] = cfg.SKIP_N_STEPS
        params['steps_per_vel'] = cfg.STEPS_PER_VEL

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
    env = utils.vec_env(cfg.env_id, norm_rew=True, num_envs=cfg.n_envs,
                        deltas=cfg.is_mod(cfg.MOD_PI_OUT_DELTAS))

    # setup model/algorithm
    training_timesteps = int(cfg.mio_steps * 1e6 * 1.05)
    learning_rate_schedule = LinearSchedule(cfg.lr_start*(1e-6), cfg.lr_final*(1e-6)).value
    clip_schedule = LinearSchedule(cfg.clip_start, cfg.clip_end).value
    network_args = {'net_arch': [{'vf': cfg.hid_layer_sizes, 'pi': cfg.hid_layer_sizes}],
                    'act_fun': tf.nn.relu} if not cfg.is_mod(cfg.MOD_CUSTOM_POLICY) else {}

    model = CustomPPO2(CustomPolicy if cfg.is_mod(cfg.MOD_CUSTOM_POLICY) else MlpPolicy,
                       env, verbose=1, n_steps=int(cfg.batch_size/cfg.n_envs),
                       policy_kwargs=network_args, nminibatches=cfg.n_mini_batches,
                       learning_rate=learning_rate_schedule, ent_coef=cfg.ent_coef,
                       gamma=cfg.gamma, noptepochs=cfg.noptepochs, cliprange_vf=cfg.cliprange,
                       cliprange=clip_schedule if cfg.is_mod(cfg.MOD_CLIPRANGE_SCHED) else cfg.cliprange,
                       tensorboard_log=cfg.save_path + 'tb_logs/')

    # init wandb
    if not cfg.DEBUG: init_wandb(model)

    # automatically launch tensorboard
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
