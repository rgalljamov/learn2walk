from os import makedirs, remove, rename
import tensorflow as tf
import numpy as np

from stable_baselines import PPO2
from scripts.common import config as cfg, utils
from stable_baselines.common.callbacks import BaseCallback

# define intervals/criteria for saving the model
EP_RETURN_THRES = 250 if not cfg.do_run() \
    else (1000 if cfg.is_mod(cfg.MOD_FLY) else 5000)
MEAN_REW_THRES = 0.05 if not cfg.do_run() else 2.5

# define evaluation interval
EVAL_MORE_FREQUENT_THRES = 3e6
EVAL_INTERVAL_BEGINNING = 250e3
EVAL_INTERVAL_FREQUENT = 100e3
EVAL_INTERVAL = EVAL_INTERVAL_BEGINNING

class TrainingMonitor(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingMonitor, self).__init__(verbose)
        # to control how often to save the model
        self.times_surpassed_ep_return_threshold = 0
        self.times_surpassed_mean_reward_threshold = 0
        # control evaluation
        self.n_steps_after_eval = EVAL_INTERVAL
        self.n_saved_models = 0
        self.mean_walked_distance = 0
        self.min_walked_distance = 0
        self.has_reached_stable_walking = False
        # log data less frequently
        self.skip_n_steps = 20
        self.skipped_steps = 20

    def _on_training_start(self) -> None:
        self.env = self.training_env

    def _on_step(self) -> bool:
        if cfg.DEBUG and self.num_timesteps > cfg.MAX_DEBUG_STEPS:
            raise SystemExit(f"Planned Exit after {cfg.MAX_DEBUG_STEPS} due to Debugging mode!")

        self.n_steps_after_eval += 1 * cfg.n_envs
        global EVAL_INTERVAL
        # skip n steps to reduce logging interval and speed up training
        if self.skipped_steps < self.skip_n_steps:
            self.skipped_steps += 1
            return True

        if self.n_steps_after_eval >= EVAL_INTERVAL and not cfg.DEBUG:
            self.n_steps_after_eval = 0
            walking_stably = self.eval_walking()
            # terminate training when stable walking has been learned
            if walking_stably:
                import wandb
                # log required num of steps to wandb
                if not self.has_reached_stable_walking:
                    wandb.run.summary['steps_to_convergence'] = self.num_timesteps
                    wandb.log({'log_steps_to_convergence': self.num_timesteps})
                    self.has_reached_stable_walking = True
                utils.log("WE COULD FINISH TRAINING EARLY!",
                          [f'Agent learned to stably walk '
                           f'after {self.num_timesteps} steps'
                           f'with mean step reward of {self.mean_reward_means}!'])
            if self.num_timesteps > EVAL_MORE_FREQUENT_THRES:
                EVAL_INTERVAL = EVAL_INTERVAL_FREQUENT

        ep_len = self.get_mean('ep_len_smoothed')
        ep_ret = self.get_mean('ep_ret_smoothed')
        mean_rew = self.get_mean('mean_reward_smoothed')

        # avoid logging data during first episode
        if ep_len < 30:
            return True

        self.log_to_tb(mean_rew, ep_len, ep_ret)
        # do not save a model if its episode length was too short
        if ep_len > 1500:
            self.save_model_if_good(mean_rew, ep_ret)

        # reset counter of skipped steps after data was logged
        self.skipped_steps = 0

        return True


    def get_mean(self, attribute_name):
        try: return np.mean(self.env.get_attr(attribute_name))
        except: return 0.333


    def log_to_tb(self, mean_rew, ep_len, ep_ret):
        moved_distance = self.get_mean('moved_distance_smooth')
        mean_ep_joint_pow_sum = self.get_mean('mean_ep_joint_pow_sum_normed_smoothed')
        mean_abs_torque_smoothed = self.get_mean('mean_abs_torque_smoothed')
        median_abs_torque_smoothed = self.get_mean('median_abs_torque_smoothed')

        # Log scalar values
        NEW_LOG_STRUCTURE = True
        if NEW_LOG_STRUCTURE:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='_distance/1. mean eval distance (deterministic)',
                                 simple_value=self.mean_walked_distance),
                tf.Summary.Value(tag='_distance/0. MIN eval distance (deterministic)',
                                 simple_value=self.min_walked_distance),
                tf.Summary.Value(tag='_distance/2. moved distance (stochastic, smoothed 0.25)',
                                 simple_value=moved_distance),
                tf.Summary.Value(tag='_distance/4. episode length (smoothed 0.75)', simple_value=ep_len),
                tf.Summary.Value(tag='_own/1. step reward (smoothed 0.25)', simple_value=mean_rew),
                tf.Summary.Value(tag='_own/2. episode return (smoothed 0.75)', simple_value=ep_ret),
                tf.Summary.Value(tag='_own/3.1. mean abs episode joint torques (smoothed 0.75)',
                                 simple_value=mean_abs_torque_smoothed),
                tf.Summary.Value(tag='_own/3.2. median abs episode joint torques (smoothed 0.75)',
                                 simple_value=median_abs_torque_smoothed),
                tf.Summary.Value(tag='_own/4. mean normed episode joint power sum (smoothed 0.75)',
                                 simple_value=mean_ep_joint_pow_sum)
            ])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            return

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='_own_data/0. mean eval distance (deterministic)', simple_value=self.mean_walked_distance),
            tf.Summary.Value(tag='_own_data/0. MIN eval distance (deterministic)', simple_value=self.min_walked_distance),
            tf.Summary.Value(tag='_own_data/1. moved distance (smoothed 0.25)', simple_value=moved_distance),
            tf.Summary.Value(tag='_own_data/2. step reward (smoothed 0.25)', simple_value=mean_rew),
            tf.Summary.Value(tag='_own_data/3. episode return (smoothed 0.75)', simple_value=ep_ret),
            tf.Summary.Value(tag='_own_data/4. episode length (smoothed 0.75)', simple_value=ep_len),
            tf.Summary.Value(tag='_own_data/5. mean normed episode joint power sum (smoothed 0.75)', simple_value=mean_ep_joint_pow_sum)
        ])
        self.locals['writer'].add_summary(summary, self.num_timesteps)


    def save_model_if_good(self, mean_rew, ep_ret):
        if cfg.DEBUG: return
        def get_mio_timesteps():
            return int(self.num_timesteps/1e6)

        ep_ret_thres = 2000 + int(EP_RETURN_THRES * (self.times_surpassed_ep_return_threshold + 1))
        if ep_ret > ep_ret_thres:
            utils.save_model(self.model, cfg.save_path,
                             'ep_ret' + str(ep_ret_thres) + f'_{get_mio_timesteps()}M')
            self.times_surpassed_ep_return_threshold += 1
            print(f'Saving model after surpassing EPISODE RETURN of {ep_ret_thres}.')
            print('Model Path: ', cfg.save_path)

        mean_rew_thres = 0.5 + MEAN_REW_THRES * (self.times_surpassed_mean_reward_threshold + 1)
        if mean_rew > (mean_rew_thres):
            utils.save_model(self.model, cfg.save_path,
                             'mean_rew' + str(int(100*mean_rew_thres)) + f'_{get_mio_timesteps()}M')
            self.times_surpassed_mean_reward_threshold += 1
            print(f'Saving model after surpassing MEAN REWARD of {mean_rew_thres}.')
            print('Model Path: ', cfg.save_path)


    def eval_walking(self):
        """
        Test the deterministic version of the current model:
        How far does it walk (in average and at least) without falling?
        @returns: If the training can be stopped as stable walking was achieved.
        """
        moved_distances, mean_rewards = [], []
        # save current model
        checkpoint = f'{int(self.num_timesteps/1e5)}'
        model_path, env_path = \
            utils.save_model(self.model, cfg.save_path, checkpoint, full=False)

        # load current model
        eval_model = PPO2.load(load_path=model_path)

        eval_env = utils.load_env(checkpoint, cfg.save_path, cfg.env_id)
        mimic_env = eval_env.venv.envs[0].env
        mimic_env.activate_evaluation()

        # evaluate deterministically
        utils.log(f'Starting model evaluation, checkpoint {checkpoint}')
        obs = eval_env.reset()
        eval_n_times = cfg.EVAL_N_TIMES if self.num_timesteps > EVAL_MORE_FREQUENT_THRES*2/3 else 10
        for i in range(eval_n_times):
            ep_dur = 0
            walked_distance = 0
            rewards = []
            while True:
                ep_dur += 1
                action, _ = eval_model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                # unnormalize reward
                reward = reward * np.sqrt(eval_env.ret_rms.var + 1e-8)
                rewards.append(reward[0])
                if done:
                    moved_distances.append(walked_distance)
                    mean_rewards.append(np.mean(rewards))
                    break
                else:
                    # we cannot get the walked distance after episode termination,
                    # as when done=True is returned, the env was already reseted.
                    walked_distance = mimic_env.data.qpos[0]

        # calculate mean walked distance
        self.mean_walked_distance = np.mean(moved_distances)
        self.min_walked_distance = np.min(moved_distances)
        # how many times 20m were reached
        runs_below_20 = np.where(np.array(moved_distances) < 20)[0]
        if eval_n_times == cfg.EVAL_N_TIMES:
            self.failed_eval_runs_indices = runs_below_20.tolist()
        self.count_stable_walks = eval_n_times - len(runs_below_20)

        ## delete evaluation model if stable walking was not achieved yet
        # or too many models were saved already
        were_enough_models_saved = self.n_saved_models >= 5
        # or walking was not human-like
        walks_humanlike = np.mean(mean_rewards) >= 0.5
        print('Mean rewards during evaluation of the deterministic model: ', mean_rewards)
        min_dist = int(self.min_walked_distance)
        mean_dist = int(self.mean_walked_distance)
        # walked 10 times at least 20 meters without falling
        has_achieved_stable_walking = min_dist > 20
        # in average stable for 20 meters but not all 20 trials were over 20m
        has_reached_high_mean_distance = mean_dist > 20
        is_stable_humanlike_walking = min_dist >= 20 and walks_humanlike
        # retain the model if it is good else delete it
        retain_model = is_stable_humanlike_walking and not were_enough_models_saved
        distances_report = [f'Min walked distance: {min_dist}m',
                            f'Mean walked distance: {mean_dist}m']
        if retain_model:
            utils.log('Saving Model:', distances_report)
            # rename model: add distances to the models names
            dists = f'_min{min_dist}mean{mean_dist}'
            new_model_path = model_path[:-4] + dists +'.zip'
            new_env_path = env_path + dists
            rename(model_path, new_model_path)
            rename(env_path, new_env_path)
            self.n_saved_models += 1
        else:
            utils.log('Deleting Model:', distances_report)
            remove(model_path)
            remove(env_path)

        return is_stable_humanlike_walking







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