"""
Loads a specified model (by path or from config) and executes it.
The policy can be used sarcastically and deterministically.
"""
import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs
from gym_mimic_envs.monitor import Monitor
from gym_mimic_envs.mimic_env import MimicEnv
from stable_baselines import PPO2
from scripts.common.utils import load_env

DETERMINISTIC_ACTIONS = False
FROM_PATH = True
RENDER = True
PATH = "/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/" \
       "dmm/deltas/pi_deltas/mim2d/16envs/ppo2/8mio/64"
if not PATH.endswith('/'): PATH += '/'
checkpoint ='ep_ret2500_6M' #'mean_rew60' # 999

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

obs = env.reset()
# env.do_fly()
env.activate_evaluation()


for i in range(10000):

    # obs, reward, done, _ = env.step(np.zeros_like(env.action_space.sample()))
    # obs, reward, done, _ = env.step(np.ones_like(env.action_space.sample()))
    # obs, reward, done, _ = env.step(env.action_space.sample())

    if FROM_PATH:
        action, hid_states = model.predict(obs, deterministic=DETERMINISTIC_ACTIONS)
        obs, reward, done, _ = vec_env.step(action)
    else:
        # follow desired trajecs
        des_qpos = env.get_ref_qpos(exclude_not_actuated_joints=True)
        obs, reward, done, _ = env.step(des_qpos)

    # only stop episode when agent has fallen
    done = env.data.qpos[1] < 0.5

    if RENDER: env.render()
    if done:
        env.reset()

env.close()
