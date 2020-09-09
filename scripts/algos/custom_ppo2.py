import time
import numpy as np

from stable_baselines import PPO2
from scripts.common.utils import log
from scripts.common import config as cfg
from scripts.behavior_cloning.dataset import get_obs_and_delta_actions

# imports required to copy the learn method
from stable_baselines import logger
from stable_baselines.common.math_util import safe_mean
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common import explained_variance, SetVerbosity, TensorboardWriter


def mirror_experiences(rollout):
    obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout
    assert obs.shape[0] == cfg.batch_size
    assert states is None
    assert len(ep_infos) == 0

    is3d = '3d' in cfg.env_name or '3pd' in cfg.env_name
    if is3d:
        # 3D Walker obs indices:
        #           0: phase, 1: des_vel, 2: com_y, 3: com_z,
        #           4: trunk_rot_x, 5: trunk_rot_y, 6: trunk_rot_z,
        #           7: hip_ang_r_sag, 8: hip_ang_r_front, 9: knee_ang_r, 10: ankle_ang_r,
        #           11: hip_ang_l_sag, 12: hip_ang_l_front 13: knee_ang_l, 14: ankle_ang_l,
        #           15: com_x_vel, 16: com_y_vel, 17:com_z_vel,
        #           18: trunk_x_ang_vel, 19: trunk_y_ang_vel, 20: trunk_z_ang_vel,
        #           21: hip_sag_vel_r, 22: hip_front_vel_r, 23: knee_vel_r, 24: ankle_vel_r,
        #           25: hip_sag_vel_l, 26: hip_front_vel_l, 27: knee_vel_l, 28: ankle_vel_l
        mirred_obs_indices = [0, 1, 2, 3,
                              4, 5, 6,
                              11, 12, 13, 14,
                              7, 8, 9, 10,
                              15, 16, 17,
                              18, 19, 20,
                              25, 26, 27, 28,
                              21, 22, 23, 24]
        # some observations have to retain the same absolute value but change the sign
        negate_obs_indices = [2, 4, 6, 16, 18, 20]
        mirred_acts_indices = [4, 5, 6, 7, 0, 1, 2, 3]
    else:
        # 2D Walker obs indices:
        #           0: phase, 1: des_vel, 2: com_z, 3: trunk_rot,
        #           4: hip_ang_r, 5: knee_ang_r, 6: ankle_ang_r,
        #           7: hip_ang_l, 8: knee_ang_l, 9: ankle_ang_l,
        #           10: com_x_vel, 11:com_z_vel, 12: trunk_ang_vel,
        #           13: hip_vel_r, 14: knee_vel_r, 15: ankle_vel_r,
        #           16: hip_vel_l, 17: knee_vel_l, 18: ankle_vel_l
        mirred_acts_indices = [3, 4, 5, 0, 1, 2]
        mirred_obs_indices = [0, 1, 2, 3, 7, 8, 9, 4, 5, 6,
                              10, 11, 12, 16, 17, 18, 13, 14, 15]

    obs_mirred = obs[:, mirred_obs_indices]
    if is3d: obs_mirred[:, negate_obs_indices] *= -1
    acts_mirred = actions[:, mirred_acts_indices]
    obs = np.concatenate((obs, obs_mirred), axis=0)
    actions = np.concatenate((actions, acts_mirred), axis=0)

    # the other values should stay the same for the mirrored experiences
    returns = np.concatenate((returns, returns))
    masks = np.concatenate((masks, masks))
    values = np.concatenate((values, values))
    neglogpacs = np.concatenate((neglogpacs, neglogpacs))
    true_reward = np.concatenate((true_reward, true_reward))

    assert true_reward.shape[0] == cfg.batch_size*2
    assert obs.shape[0] == cfg.batch_size*2

    return obs, returns, masks, actions, values, \
           neglogpacs, states, ep_infos, true_reward


def generate_experiences_from_refs(rollout, ref_obs, ref_acts):
    """
    Generate experiences from reference trajectories.
    - obs and actions can be used without a change. TODO: obs and acts should be normalized by current running stats
    - predicted state values are estimated as the mean value of taken experiences. TODO: query VF network
    - neglogpacs, -log[p(a|s)], are estimated by using the smallest probability of taken experiences. TODO: query PI network
    - returns are estimated by max return of taken (s,a)-pairs
    TODO: Mirror refs
    """

    obs, returns, masks, actions, values, neglogpacs, \
    states, ep_infos, true_reward = rollout

    n_ref_exps = ref_obs.shape[0]
    ref_returns = np.ones((n_ref_exps,), dtype=np.float32) * np.max(returns)
    ref_values = np.ones((n_ref_exps,), dtype=np.float32) * np.mean(values)
    ref_masks = np.array([False] * n_ref_exps)
    ref_neglogpacs = np.ones_like(ref_values) * np.mean(neglogpacs)

    obs = np.concatenate((obs, ref_obs), axis=0)
    actions = np.concatenate((actions, ref_acts), axis=0)
    returns = np.concatenate((returns, ref_returns))
    masks = np.concatenate((masks, ref_masks))
    values = np.concatenate((values, ref_values))
    neglogpacs = np.concatenate((neglogpacs, ref_neglogpacs))

    return obs, actions, returns, masks, values, neglogpacs

class CustomPPO2(PPO2):
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        log('Using CustomPPO2!')

        self.mirror_experiences = cfg.is_mod(cfg.MOD_MIRROR_EXPS)

        if cfg.is_mod(cfg.MOD_REFS_REPLAY) or cfg.is_mod(cfg.MOD_ONLINE_CLONE):
            # load obs and actions generated from reference trajectories
            self.ref_obs, self.ref_acts = get_obs_and_delta_actions(norm_obs=True, norm_acts=True, fly=False)

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
                batch_size = self.n_batch // self.nminibatches
                if self.n_batch % self.nminibatches != 0:
                    log("The number of minibatches (`nminibatches`) "
                        "is not a factor of the total number of samples "
                        "collected per rollout (`n_batch`), "
                        "some samples won't be used.")
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                callback.on_rollout_start()

                # try getting rollout 3 times
                tried_rollouts = 0
                while tried_rollouts < 3:
                    try:
                        # true_reward is the reward without discount
                        rollout = self.runner.run(callback)
                        break
                    except Exception as ex:
                        log(f'Rollout failed {tried_rollouts+1} times!'
                            f'Catched exception: {ex}')
                        tried_rollouts += 1
                        time.sleep(10*tried_rollouts)

                # reset count once, rollout was successful
                tried_rollouts = 0


                # Unpack
                if self.mirror_experiences:
                    obs, returns, masks, actions, values, neglogpacs, \
                    states, ep_infos, true_reward = mirror_experiences(rollout)
                else:
                    obs, returns, masks, actions, values, neglogpacs, \
                    states, ep_infos, true_reward = rollout

                if np.random.randint(low=0, high=9, size=1)[0] == 7:
                    log(f'Values and Returns of collected experiences: ',
                    [f'min returns:\t{np.min(returns)}', f'min values:\t\t{np.min(values)}',
                     f'mean returns:\t{np.mean(returns)}', f'mean values:\t{np.mean(values)}',
                     f'max returns:\t{np.max(returns)}', f'max values:\t\t{np.max(values)}'])

                if cfg.is_mod(cfg.MOD_REFS_REPLAY) or cfg.is_mod(cfg.MOD_ONLINE_CLONE):
                    # load ref experiences and treat them as real experiences
                    obs, actions, returns, masks, values, neglogpacs = \
                        generate_experiences_from_refs(rollout, self.ref_obs, self.ref_acts)

                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    n_epochs = self.noptepochs \
                        if not cfg.is_mod(cfg.MOD_ONLINE_CLONE) or update > 9 else 200
                    for epoch_num in range(n_epochs):
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
