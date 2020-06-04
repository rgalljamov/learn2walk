'''
Interface for environments using reference trajectories.
'''
import gym, mujoco_py

class MimicEnv:
    def __init__(self: gym.Env, ref_trajecs):
        '''@param: self: gym environment implementing the MimicEnv interface.'''
        self.refs = ref_trajecs

    def playback_ref_trajectories(self, timesteps=1000):
        self.reset()
        sim = self.sim
        for i in range(timesteps):
            old_state = sim.get_state()
            new_state = mujoco_py.MjSimState(old_state.time,
                                             self.refs.get_qpos(i),
                                             self.refs.get_qvel(i),
                                             old_state.act, old_state.udd_state)
            sim.set_state(new_state)
            sim.forward()
            self.render()

    def get_qpos(self, timestep):
        raise NotImplementedError

