import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs

env = gym.make('MimicWalker3d-v0')
env.playback_ref_trajectories()
env.close()
