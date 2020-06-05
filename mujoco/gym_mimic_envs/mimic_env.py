'''
Interface for environments using reference trajectories.
'''
import gym, mujoco_py
import numpy as np
from scripts.common.ref_trajecs import ReferenceTrajectories as RefTrajecs

# Workaround: MujocoEnv calls step() before calling reset()
# Then, RSI is not executed yet and ET gets triggered during step()
_rsinitialized = False

class MimicEnv:
    def __init__(self: gym.Env, ref_trajecs:RefTrajecs):
        '''@param: self: gym environment implementing the MimicEnv interface.'''
        self.refs = ref_trajecs

    def get_joint_kinematics(self):
        '''Returns qpos and qvel of the agent.'''
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return qpos, qvel

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


    def is_early_termination(self, max_dev_pos=0.25, max_dev_vel=0.25):
        '''Early Termination:
           @returns: True if qpos and qvel have deviated
           too much from the reference trajectories.
           @params: max_dev_x are both percentages of maximum range
                    of the corresponding joint ref trajectories.'''

        if not _rsinitialized:
            # ET only works after RSI was executed
            return False

        qpos, qvel = self.get_joint_kinematics()
        ref_qpos, ref_qvel = self.refs.get_ref_kinmeatics()
        pos_ranges, vel_ranges = self.refs.get_kinematic_ranges()
        delta_pos = np.abs(ref_qpos - qpos)
        delta_vel = np.abs(ref_qvel - qvel)

        # was the maximum allowed deviation exceeded
        pos_exceeded = delta_pos > max_dev_pos * pos_ranges
        vel_exceeded = delta_vel > max_dev_vel * vel_ranges

        # investigate deviations:
        # which kinematics have exceeded the allowed deviations
        pos_is = np.where(pos_exceeded==True)[0]
        vel_is = np.where(vel_exceeded==True)[0]
        pos_labels, vel_labels = self.refs.get_labels_by_index(pos_is, vel_is)

        print()
        for i_pos, pos_label in enumerate(pos_labels):
            print(f"{pos_label} \t exceeded max range of {pos_ranges[i_pos]} "
                  f"by {delta_pos[i_pos]} after {self.refs.ep_dur} steps")

        print()
        for i_vel, vel_label in enumerate(vel_labels):
            print(f"{vel_label} \t exceeded max range of {vel_ranges[i_vel]} "
                  f"by {delta_vel[i_vel]} after {self.refs.ep_dur} steps")


        return pos_exceeded.any() or vel_exceeded.any()


