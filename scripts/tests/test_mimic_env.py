import gym
import gym_mimic_envs

env = gym.make('MimicWalker2d-v0')
env.reset()

for _ in range(1000):
    env.step(env.action_space.sample())
    env.render()

env.close()
