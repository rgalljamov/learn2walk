import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs
from gym_mimic_envs.monitor import Monitor
import numpy as np

env = gym.make('MimicWalker2d-v0')
env = Monitor(env)
# env.playback_ref_trajectories(10000, pd_pos_control=True)
env.reset()

for i in range(10000):

    # obs, reward, done, _ = env.step(np.zeros_like(env.action_space.sample()))
    # obs, reward, done, _ = env.step(np.ones_like(env.action_space.sample()))
    # obs, reward, done, _ = env.step(env.action_space.sample())

    # follow desired trajecs
    des_qpos = env.get_ref_qpos(exclude_not_actuated_joints=True)
    obs, reward, done, _ = env.step(des_qpos)

    # env.render()
    if done:
        env.reset()

env.close()
