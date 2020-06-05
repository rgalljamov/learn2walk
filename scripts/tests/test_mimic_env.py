import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs
import numpy as np

env = gym.make('MimicWalker2d-v0')
# env.playback_ref_trajectories(1000)
env.reset()

for i in range(10000):

    # obs, reward, done, _ = env.step(np.zeros_like(env.action_space.sample()))
    obs, reward, done, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        env.reset()

env.close()
