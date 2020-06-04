import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs

env = gym.make('MimicWalker2d-v0')
env.playback_ref_trajectories(10000)
env.close()
