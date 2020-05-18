import os.path
import numpy as np
from matplotlib import pyplot as plt
from scripts.common import utils
from scripts.common import config as cfg

# to decrease the amount of deprecation warnings
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from stable_baselines import PPO2

RENDER = False
PLOT_RESULTS = True

# which model should be evaluated
run_id = 206
checkpoint = cfg.final_checkpoint

# evaluate for n episodes
n_eps = 100
# how many actions to record in each episode
rec_n_steps = 1000

def eval_model():
    """ evaluate model specified in the config file """
    utils.config_plots()

    print('\n---------------------------\n'
              'MODEL EVALUATION STARTED'
          '\n---------------------------\n')

    # change save_path to specified model
    save_path = cfg.save_path_norun + f'{run_id}/'

    print('\nModel:\n', save_path + '\n')

    # load model
    model = PPO2.load(load_path=save_path + f'models/model_{checkpoint}.zip')

    # load a single environment for evaluation
    # todo: we could also use multiple envs to speedup eval
    env = utils.vec_env(cfg.env_id, num_envs=1, norm_rew=False)
    # set the calculated running means for obs and rets
    env.load_running_average(save_path + f'envs/env_{checkpoint}')


    ep_rewards, all_returns, ep_durations = [], [], []
    all_rewards = np.zeros((n_eps, rec_n_steps))
    all_actions = np.zeros((n_eps, 4, rec_n_steps))

    ep_count, ep_dur = 0,0
    obs = env.reset()

    if not RENDER: print('Episodes finished:\n0 ', end='')

    while True:
        ep_dur += 1
        action, _states = model.predict(obs, deterministic=True)
        if ep_dur <= rec_n_steps:
            all_actions[ep_count, :, ep_dur - 1] = action
        # time.sleep(0.025)
        obs, reward, done, info = env.step(action)
        ep_rewards += [reward[0]]
        if done.any():
            ep_durations.append(ep_dur)
            # clip ep_dur to max number of steps to save
            ep_dur = min(ep_dur, rec_n_steps)
            all_rewards[ep_count,:ep_dur] = np.asarray(ep_rewards)[:ep_dur]
            ep_return = np.sum(ep_rewards)
            all_returns.append(ep_return)
            if RENDER: print('ep_return : ', ep_return)
            # reset all episode specific containers and counters
            ep_rewards = []
            ep_dur = 0
            ep_count += 1
            # stop evaluation after enough episodes were observed
            if ep_count >= n_eps: break
            elif ep_count % 10 == 0:
                print(f'-> {ep_count}', end=' ', flush=True)
        if RENDER: env.render()
    env.close()
    print()

    mean_return = np.mean(all_returns)
    print('\nAverage episode return was: ', mean_return)

    # create the metrics folder
    metrics_path = save_path + f'metrics/model_{checkpoint}/'

    if not os.path.exists(metrics_path + '/'):
        os.makedirs(metrics_path)

    # create folder for metrics of a certain checkpoint
    # if not os.path.exists(metrics_path + ):
    #     os.makedirs(metrics_path)

    np.save(metrics_path + '/{}_mean_ret_on_{}eps'.format(int(mean_return), n_eps), mean_return)
    np.save(metrics_path + '/rewards', all_rewards)
    np.save(metrics_path + '/actions', all_actions)
    np.save(metrics_path + '/ep_returns', all_returns)

    # count and save number of parameters in the model
    num_policy_params = np.sum([np.prod(tensor.get_shape().as_list())
                                for tensor in model.params if 'pi' in tensor.name])
    num_valfunc_params = np.sum([np.prod(tensor.get_shape().as_list())
                                for tensor in model.params if 'vf' in tensor.name])
    num_params = np.sum([np.prod(tensor.get_shape().as_list())
                         for tensor in model.params])

    count_params_dict = {'n_pi_params': num_policy_params, 'n_vf_params':num_valfunc_params,
                         'n_params': num_params}

    np.save(metrics_path + '/weights_count', count_params_dict)
    print(count_params_dict)

    if PLOT_RESULTS:
        plt.plot(all_returns)
        plt.title(f"Returns of {n_eps} epochs")
        plt.show()


if __name__ == "__main__":
    eval_model()
# eval.py was called from another script
else:
    RENDER = False