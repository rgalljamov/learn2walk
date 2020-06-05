'''
Interface for environments using reference trajectories.
'''
import gym, mujoco_py
from scripts.common.ref_trajecs import ReferenceTrajectories as RefTrajecs

class MimicEnv:
    def __init__(self: gym.Env, ref_trajecs:RefTrajecs):
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

        self.close()
        raise SystemExit('Environment intentionally closed after playing back trajectories.')


    def _get_obs(self):
        qpos, qvel = self.get_joint_kinematics()
        return np.concatenate([qpos, qvel]).ravel()


    def reset_model(self):
        '''WARNING: This method seems to be specific to MujocoEnv.
           Other gym environments just use reset().'''
        qpos, qvel = self.get_random_init_state()
        self.set_state(qpos, qvel)
        return self._get_obs()


    def get_random_init_state(self):
        ''' Random State Initialization:
            @returns: qpos and qvel of a random step at a random position'''
        global _rsinitialized
        _rsinitialized = True
        return self.refs.get_random_init_state()


