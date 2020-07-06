import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs
from gym_mimic_envs.monitor import Monitor
from gym_mimic_envs.mimic_env import MimicEnv
from stable_baselines import PPO2
from scripts.common.utils import load_env
import numpy as np

DETERMINISTIC_ACTIONS = False
FROM_PATH = False
RENDER = True
PATH = "/home/rustam/code/remote/models/deepmim/500Nm/et25/mim_walker2d/" \
       "8envs/ppo2/hyper_dflt/10mio/ent-250_lr500to1_clp1_bs8_imrew613_pun100_gamma999/312"
if not PATH.endswith('/'): PATH += '/'
checkpoint = 'ep_ret1500' # 999

if FROM_PATH:
    # load model
    model_path = PATH + f'models/model_{checkpoint}.zip'
    model = PPO2.load(load_path=model_path)

    print('\nModel:\n', model_path + '\n')

    env = load_env(checkpoint, PATH)
    # env = Monitor(env, True)

else:
    env = gym.make('MimicWalker2d-v0')
    env = Monitor(env)
    # env.playback_ref_trajectories(10000, pd_pos_control=True)

obs = env.reset()
if not isinstance(env, Monitor):
    # VecNormalize wrapped DummyVecEnv
    vec_env = env
    env = env.venv.envs[0]

env.do_fly()
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
