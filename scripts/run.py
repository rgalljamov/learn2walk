"""
Loads a specified model (by path or from config) and executes it.
The policy can be used sarcastically and deterministically.
"""
import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs
from gym_mimic_envs.monitor import Monitor
from gym_mimic_envs.mujoco.mimic_walker2d import MimicWalker2dEnv
from stable_baselines import PPO2
from scripts.common.utils import load_env
from scripts.common import config as cfg

# paths
# PD baseline
path_pd_baseline = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/' \
                   'cstm_pi/mim3d/8envs/ppo2/16mio/918-evaled-ret71'
path_pd_normed_deltas = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/' \
                        'pi_deltas/norm_acts/cstm_pi/mim3d/8envs/ppo2/16mio/431-evaled-ret81'
path_trq_baseline = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/' \
                    'cstm_pi/mim_trq_ff3d/8envs/ppo2/8mio/296-evaled-ret79'
path_trq_dif_imitation = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/' \
                         'dmm/cstm_pi/mim_trq_ff3d/8envs/ppo2/16mio/580'


FLY = False
DETERMINISTIC_ACTIONS = True
RENDER = True

SPEED_CONTROL = False

FROM_PATH = True
# WARNING: Besides changing the paths,
# also the approach modification in config.py and the environment has to be changed!
PATH = path_trq_baseline # path_pd_baseline # path_pd_normed_deltas #
    # "/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/" \
    #    "cstm_pi/mim_trq3d/8envs/ppo2/16mio/658-evaled-ret78"
PATH = path_trq_dif_imitation
if not PATH.endswith('/'): PATH += '/'
checkpoint = '37_min22mean22' #'mean_rew80_7M' # '39_min22mean22' #'final' # '33_min24mean24' # 'ep_ret2000_7M' #'mean_rew60'

if FLY: cfg.rew_weights = "6400"

if FROM_PATH:
    # check if correct reference trajectories are used
    if cfg.MOD_REFS_RAMP in PATH and not cfg.is_mod(cfg.MOD_REFS_RAMP):
        raise AssertionError('Model trained on ramp-trajecs but is used with constant speed trajecs!')
    # load model
    model_path = PATH + f'models/model_{checkpoint}.zip'
    model = PPO2.load(load_path=model_path)

    print('\nModel:\n', model_path + '\n')

    env = load_env(checkpoint, PATH, cfg.env_id)

else:
    env = gym.make(cfg.env_id)
    env = Monitor(env)
    # env.playback_ref_trajectories(10000, pd_pos_control=True)

if not isinstance(env, Monitor):
    # VecNormalize wrapped DummyVecEnv
    vec_env = env
    env = env.venv.envs[0]

if SPEED_CONTROL:
    env.activate_speed_control([0.8, 1.25])

obs = env.reset()
if FLY: env.do_fly()
env.activate_evaluation()

for i in range(10000):

    # obs, reward, done, _ = env.step(np.zeros_like(env.action_space.sample()))
    # obs, reward, done, _ = env.step(np.ones_like(env.action_space.sample()))
    # obs, reward, done, _ = env.step(env.action_space.sample())

    if FROM_PATH:
        action, hid_states = model.predict(obs, deterministic=DETERMINISTIC_ACTIONS)
        obs, reward, done, _ = vec_env.step(action)
    else:
        # save qpos of actuated joints for reward calculation
        actq_pos_before_step = env.get_qpos(True, True)

        # follow desired trajecs with PD Position Controllers
        des_qpos = env.get_ref_qpos(exclude_not_actuated_joints=True)

        if cfg.is_mod(cfg.MOD_PI_OUT_DELTAS):
            des_qpos -= actq_pos_before_step
            # rescale actions to [-1,1]
            if cfg.is_mod(cfg.MOD_NORM_ACTS):
                des_qpos /= env.get_max_qpos_deltas()

        obs, reward, done, _ = env.step(des_qpos)

    # only stop episode when agent has fallen
    done = env.data.qpos[env.env._get_COM_indices()[-1]] < 0.5
    # done = i % 300 == 0

    if RENDER: env.render()
    if done:
        env.reset()

env.close()
