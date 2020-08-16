"""
Loads a specified model (by path or from config) and executes it.
The policy can be used sarcastically and deterministically.
"""
import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs
from gym_mimic_envs.monitor import Monitor
from stable_baselines import PPO2
from scripts.common.utils import load_env
from scripts.common import config as cfg

FLY = False
DETERMINISTIC_ACTIONS = True
RENDER = False

SPEED_CONTROL = False

FROM_PATH = True
PATH = "/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/" \
       "refs_ramp/normd_com_vel/cstm_pi/pi_deltas/norm_acts/mim2d/8envs/ppo2/16mio/719-evaled"
if not PATH.endswith('/'): PATH += '/'
checkpoint = 999 # 'ep_ret2000_7M' #'mean_rew60'

if FLY: cfg.rew_weights = "6400"

if FROM_PATH:
    # load model
    model_path = PATH + f'models/model_{checkpoint}.zip'
    model = PPO2.load(load_path=model_path)

    print('\nModel:\n', model_path + '\n')

    env = load_env(checkpoint, PATH)

else:
    env = gym.make('MimicWalker2d-v0')
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
    done = env.data.qpos[1] < 0.5

    if RENDER: env.render()
    if done:
        env.reset()

env.close()
