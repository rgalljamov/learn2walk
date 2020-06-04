import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs

env = gym.make('MimicWalker3d-v0')
env.reset()

refs = env.refs

for i in range(10000):

    # env.step(env.action_space.sample())

    # obs, reward, done, _ = env.step(np.zeros_like(env.action_space.sample()))

    data = env.sim.data
    qpos = data.qpos
    qvel = data.qvel

    old_state = env.sim.get_state()
    new_state = mujoco_py.MjSimState(old_state.time,
                                     refs.get_by_indices(range(15),i),
                                     refs.get_by_indices(range(15,29),i),
                                     old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)
    env.sim.forward()



    env.render()



env.close()
