import time
import numpy as np

from stable_baselines import PPO2
from scripts.common.utils import log
from scripts.common import config as cfg

# imports required to copy the learn method
from stable_baselines import logger
from stable_baselines.common.math_util import safe_mean
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common import explained_variance, SetVerbosity, TensorboardWriter


def mirror_experiences(rollout):
    obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout
    assert obs.shape == (cfg.batch_size, 19)
    assert actions.shape == (cfg.batch_size, 6)
    assert states is None
    assert len(ep_infos) == 0
    # obs indices: 0: phase, 1: des_vel, 2: com_z, 3: trunk_rot,
    #              4: hip_ang_r, 5: knee_ang_r, 6: ankle_ang_r,
    #              7: hip_ang_l, 8: knee_ang_l, 9: ankle_ang_l,
    #              10: com_x_vel, 11:com_z_vel, 12: trunk_ang_vel,
    #              13: hip_vel_r, 14: knee_vel_r, 15: ankle_vel_r,
    #              16: hip_vel_l, 17: knee_vel_l, 18: ankle_vel_l
    mirred_obs_indices = [0, 1, 2, 3, 7, 8, 9, 4, 5, 6,
                          10, 11, 12, 16, 17, 18, 13, 14, 15]
    obs_mirred = obs[:, mirred_obs_indices]
    acts_mirred = actions[:, [3, 4, 5, 0, 1, 2]]
    obs = np.concatenate((obs, obs_mirred), axis=0)
    actions = np.concatenate((actions, acts_mirred), axis=0)

    # the other values should stay the same for the mirrored experiences
    returns = np.concatenate((returns, returns))
    masks = np.concatenate((masks, masks))
    values = np.concatenate((values, values))
    neglogpacs = np.concatenate((neglogpacs, neglogpacs))
    true_reward = np.concatenate((true_reward, true_reward))

    assert true_reward.shape[0] == cfg.batch_size*2
    assert obs.shape == (cfg.batch_size*2, 19)

    return obs, returns, masks, actions, values, \
           neglogpacs, states, ep_infos, true_reward


class CustomPPO2(PPO2):
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        log('Using CustomPPO2!')

        self.mirror_experiences = cfg.is_mod(cfg.MOD_MIRROR_EXPS)

        super(CustomPPO2, self).__init__(policy, env, gamma, n_steps, ent_coef, learning_rate, vf_coef,
                                         max_grad_norm, lam, nminibatches, noptepochs, cliprange, cliprange_vf,
                                         verbose, tensorboard_log, _init_setup_model, policy_kwargs,
                                         full_tensorboard_log, seed, n_cpu_tf_sess)

    # ----------------------------------
    # OVERWRITTEN CLASSES
    # ----------------------------------

    def setup_model(self):
        """ Overwritten to double the batch size when experiences are mirrored. """
        super(CustomPPO2, self).setup_model()
        if self.mirror_experiences:
            log('Mirroring observations and actions to improve sample-efficiency.')
            self.n_batch *= 2

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        """
        Just copied from the stable_baselines.ppo2 implementation.
        Goal is to change some parts of it later.
        """
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            t_first_start = time.time()
            n_updates = total_timesteps // self.n_batch

            callback.on_training_start(locals(), globals())

            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                               "is not a factor of the total number of samples "
                                                               "collected per rollout (`n_batch`), "
                                                               "some samples won't be used."
                                                               )
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                # Unpack
                if self.mirror_experiences:
                    obs, returns, masks, actions, values, neglogpacs, \
                    states, ep_infos, true_reward = mirror_experiences(rollout)
                else:
                    obs, returns, masks, actions, values, neglogpacs, \
                    states, ep_infos, true_reward = rollout

                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states,
                                                                 cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    if self.mirror_experiences: self.n_steps = int(self.n_steps * 2)
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)
                    if self.mirror_experiences: self.n_steps = int(self.n_steps / 2)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

            callback.on_training_end()
            return self
