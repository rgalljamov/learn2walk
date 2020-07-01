'''
Interface for environments using reference trajectories.
'''
import gym, mujoco_py
import numpy as np
import seaborn as sns
from scripts.common.utils import log
from scripts.common.ref_trajecs import ReferenceTrajectories as RefTrajecs


# Workaround: MujocoEnv calls step() before calling reset()
# Then, RSI is not executed yet and ET gets triggered during step()
_rsinitialized = False

# flag if ref trajectories are played back
_play_ref_trajecs = False

class MimicEnv:

    def __init__(self: gym.Env, ref_trajecs:RefTrajecs):
        '''@param: self: gym environment implementing the MimicEnv interface.'''
        self.refs = ref_trajecs
        # adjust simulation properties to the frequency of the ref trajec
        self.model.opt.timestep = 1e-3
        self.frame_skip = 5
        # names of all robot kinematics
        self.kinem_labels = self.refs.get_kinematics_labels()
        # keep the body in the air for testing purposes
        self._FLY = False


    def step(self):
        """
        Returns
        -------
        True if MimicEnv was already instantiated.
        Workaround (see doc string for _rsinitialized)
        """
        if not _rsinitialized:
            return False

        if self.refs is None:
            log("MimicEnv.step() called before refs were initialized!")
            return False

        self.refs.next()
        return True


    def get_joint_kinematics(self, exclude_com=False, concat=False):
        '''Returns qpos and qvel of the agent.'''
        qpos = np.copy(self.sim.data.qpos)
        qvel = np.copy(self.sim.data.qvel)
        if exclude_com:
            qpos = self._remove_by_indices(qpos, self._get_COM_indices())
            qvel = self._remove_by_indices(qvel, self._get_COM_indices())
        if concat:
            return np.concatenate([qpos, qvel]).flatten()
        return qpos, qvel

    def get_qpos(self):
        return np.copy(self.sim.data.qpos)

    def get_qvel(self):
        return np.copy(self.sim.data.qvel)

    def get_ref_qpos(self, exclude_com=False, exclude_not_actuated_joints=False):
        qpos = self.refs.get_qpos()
        if exclude_not_actuated_joints:
            qpos = self._remove_by_indices(qpos, self._get_not_actuated_joint_indices())
        elif exclude_com:
            qpos = self._remove_by_indices(qpos, self._get_COM_indices())
        return qpos

    def get_ref_qvel(self, exclude_com=False, exclude_not_actuated_joints=False):
        qvel = self.refs.get_qvel()
        if exclude_not_actuated_joints:
            qvel = self._remove_by_indices(qvel, self._get_not_actuated_joint_indices())
        elif exclude_com:
            qvel = self._remove_by_indices(qvel, self._get_COM_indices())
        return qvel

    def get_joint_torques(self):
        return np.copy(self.sim.data.actuator_force)

    def playback_ref_trajectories(self, timesteps=2000, pd_pos_control=False):
        global _play_ref_trajecs
        _play_ref_trajecs = True

        self.reset()
        if pd_pos_control:
            for i in range(timesteps):
                # hold com x and z position and trunk rotation constant
                FLIGHT = self._FLY
                ignore_not_actuated_joints = True and not FLIGHT

                if not ignore_not_actuated_joints:
                    # position servos do not actuate all joints
                    # for those joints, we still have to set the joint positions by hand
                    ref_qpos = self.get_ref_qpos()
                    ref_qvel = self.get_ref_qvel()
                    qpos = self.get_qpos()
                    qvel = self.get_qvel()

                    if FLIGHT:
                        ref_qpos[[0,1,2]] = [0, 1.5, 0]
                        qvel[[0,1]] = 0
                    # set the not actuated joint position from refs
                    not_actuated_is = self._get_not_actuated_joint_indices()
                    qpos[not_actuated_is] = ref_qpos[not_actuated_is]
                    qvel[not_actuated_is] = ref_qvel[not_actuated_is]
                    self.set_joint_kinematics_in_sim(qpos, qvel)

                    # self.sim.data.qvel[:] = 0
                    # fix all other joints to focus tuning control gains of a single joint pair
                    # self.sim.data.qpos[[0, 2, 3, 4, 6, 7]] = 0
                    # self.sim.data.qvel[[0, 2, 3, 4, 6, 7]] = 0

                # follow desired trajecs
                des_qpos = self.get_ref_qpos(exclude_not_actuated_joints=True)
                # des_qpos[0] = 0
                # des_qpos[1] = 0.4
                # if i % 60 == 0:
                #     self.reset()
                if FLIGHT:
                    self.sim.data.qpos[[0, 1, 2]] = [0, 1.5, 0]
                    self.sim.data.qvel[[0, 1, 2]] = 0
                obs, reward, done, _ = self.step(des_qpos)
                # obs, reward, done, _ = self.step(np.ones_like(des_qpos)*(0))
                # ankle tune:
                # obs, reward, done, _ = self.step([0, 0, -0.3, 0, 0, -0.3])
                # knee tune:
                # obs, reward, done, _ = self.step([0, 0.2, 0, 0, 0.2, 0])
                # obs, reward, done, _ = self.step([-0.5, 1, -.2, -0.5, 1, -.2])
                self.render()
                if done:
                    self.reset()
        else:
            for i in range(timesteps):
                self.refs.next()
                ZERO_INPUTS = False
                if ZERO_INPUTS:
                    qpos, qvel = self.get_joint_kinematics()
                    qpos= np.zeros_like(qpos)
                    qvel = np.zeros_like(qvel)
                    self.set_joint_kinematics_in_sim(qpos, qvel)
                else:
                    self.set_joint_kinematics_in_sim()
                # self.step(np.zeros_like(self.action_space.sample()))
                self.sim.forward()
                self.render()

        _play_ref_trajecs = False
        self.close()
        raise SystemExit('Environment intentionally closed after playing back trajectories.')

    def set_joint_kinematics_in_sim(self, qpos=None, qvel=None):
        old_state = self.sim.get_state()
        if qpos is None:
            qpos, qvel = self.refs.get_ref_kinmeatics()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)

    def _get_obs(self):
        global _rsinitialized
        qpos, qvel = self.get_joint_kinematics()
        if _rsinitialized:
            desired_walking_speed = self.refs.get_step_velocity()
            phase = self.refs.get_phase_variable()
        else:
            desired_walking_speed = -3.33
            phase = 0
        obs = np.concatenate([np.array([phase, desired_walking_speed]), qpos, qvel]).ravel()
        return obs

    def reset_model(self):
        '''WARNING: This method seems to be specific to MujocoEnv.
           Other gym environments just use reset().'''
        qpos, qvel = self.get_random_init_state()

        # # tune hip pd gains
        # left_hip_index = 6
        # qpos_hip_left = qpos[left_hip_index]
        # qvel_hip_left = qvel[left_hip_index]
        # qpos = np.zeros_like(qpos)
        # qvel = np.zeros_like(qvel)
        # qpos[left_hip_index] = qpos_hip_left
        # qvel[left_hip_index] = qvel_hip_left
        # qpos[3] = 0.1

        # # tune knee pd gains
        # left_index = 7
        # right_index = 4
        # left_joint_pos = qpos[left_index]
        # left_joint_vel = qvel[left_index]
        # qpos[left_index] = left_joint_pos
        # qvel[left_index] = left_joint_vel
        # qpos[right_index] = 0.5
        # qvel[right_index] = 0
        # # set ankle pos and vel to zero
        # ankle_ins = [5,8]
        # qpos[ankle_ins] = 0
        # qvel[ankle_ins] = 0

        ### avoid huge joint toqrues from PD servos after RSI
        # Explanation: on reset, ctrl is set to all zeros.
        # When we set a desired state during RSI, we suddenly change the current state.
        # Without also changing the target angles, there will be a huge difference
        # between target and current angles and PD Servos will kick in with high torques
        self.data.ctrl[:] = self._remove_by_indices(qpos, self._get_not_actuated_joint_indices())

        self.set_state(qpos, qvel)
        rew = self.get_imitation_reward()
        assert rew > 0.95 if not self._FLY else 0.5, \
            f"Reward should be around 1 after RSI, but was {rew}!"
        assert not self.has_exceeded_allowed_deviations()
        return self._get_obs()


    def get_random_init_state(self):
        ''' Random State Initialization:
            @returns: qpos and qvel of a random step at a random position'''
        global _rsinitialized
        _rsinitialized = True
        return self.refs.get_random_init_state()


    def get_ref_kinematics(self, exclude_com=False, concat=False):
        qpos, qvel = self.refs.get_ref_kinmeatics()
        if exclude_com:
            qpos = self._remove_by_indices(qpos, self._get_COM_indices())
            qvel = self._remove_by_indices(qvel, self._get_COM_indices())
        if concat:
            return np.concatenate([qpos, qvel]).flatten()
        return qpos, qvel


    def get_pose_reward(self):
        # get sim and ref joint positions excluding com position
        qpos, _ = self.get_joint_kinematics(exclude_com=True)
        ref_pos, _ = self.get_ref_kinematics(exclude_com=True)
        if self._FLY:
            ref_pos[0] = 0
        dif = qpos - ref_pos
        dif_sqrd = np.square(dif)
        sum = np.sum(dif_sqrd)
        pose_rew = np.exp(-2 * sum)
        return pose_rew

    def get_vel_reward(self):
        _, qvel = self.get_joint_kinematics(exclude_com=True)
        _, ref_vel = self.get_ref_kinematics(exclude_com=True)
        if self._FLY:
            ref_vel[0] = 0
        dif = qvel - ref_vel
        dif_sqrd = np.square(dif)
        sum = np.sum(dif_sqrd)
        vel_rew = np.exp(-0.1 * sum)
        return vel_rew

    def get_com_reward(self):
        qpos, qvel = self.get_joint_kinematics()
        ref_pos, ref_vel = self.get_ref_kinematics()
        com_is = self._get_COM_indices()
        com_pos, com_ref = qpos[com_is], ref_pos[com_is]
        dif = com_pos - com_ref
        dif_sqrd = np.square(dif)
        sum = np.sum(dif_sqrd)
        com_rew = np.exp(-10 * sum)
        return com_rew

    def _remove_by_indices(self, list, indices):
        """
        Removes specified indices from the passed list and returns it.
        """
        new_list = [item for i, item in enumerate(list) if i not in indices]
        return np.array(new_list)

    def get_imitation_reward(self):
        global _rsinitialized
        if not _rsinitialized:
            return -3.33
        # todo: do we need the end-effector reward?
        w_pos, w_vel, w_com = 0.5, 0.1, 0.4
        pos_rew = self.get_pose_reward()
        vel_ref = self.get_vel_reward()
        com_rew = self.get_com_reward()
        imit_rew = w_pos * pos_rew + w_vel * vel_ref + w_com * com_rew
        return imit_rew

    def do_terminate_early(self, rew, com_height, trunk_ang_saggit,
                           rew_threshold = 0.05):
        """
        Early Termination based on reward, falling and episode duration
        """
        if (not _rsinitialized) or _play_ref_trajecs:
            # ET only works after RSI was executed
            return False

        # calculate if allowed com height was exceeded (e.g. walker felt down)
        com_height_des = self.refs.get_com_height()
        com_delta = np.abs(com_height_des - com_height)
        com_deviation_prct = com_delta/com_height_des
        allowed_com_deviation_prct = 0.4
        com_max_dev_exceeded = com_deviation_prct > allowed_com_deviation_prct
        if self._FLY: com_max_dev_exceeded = False

        # calculate if trunk angle exceeded limits of 45° (0.785 in rad)
        trunk_ang_exceeded = np.abs(trunk_ang_saggit) > 0.7

        rew_too_low = rew < rew_threshold
        max_episode_dur_reached = self.refs.ep_dur > 10000
        return com_max_dev_exceeded or trunk_ang_exceeded or rew_too_low or max_episode_dur_reached

    def has_exceeded_allowed_deviations(self, max_dev_pos=0.5, max_dev_vel=2):
        '''Early Termination based on trajectory deviations:
           @returns: True if qpos and qvel have deviated
           too much from the reference trajectories.
           @params: max_dev_x are both percentages of maximum range
                    of the corresponding joint ref trajectories.'''

        if not _rsinitialized:
            # ET only works after RSI was executed
            return False

        qpos, qvel = self.get_joint_kinematics()
        ref_qpos, ref_qvel = self.get_ref_kinematics()
        pos_ranges, vel_ranges = self.refs.get_kinematic_ranges()
        # increase trunk y rotation range
        pos_ranges[2] *= 5
        vel_ranges[2] *= 5
        delta_pos = np.abs(ref_qpos - qpos)
        delta_vel = np.abs(ref_qvel - qvel)

        # was the maximum allowed deviation exceeded
        pos_exceeded = delta_pos > max_dev_pos * pos_ranges
        vel_exceeded = delta_vel > max_dev_vel * vel_ranges

        # investigate deviations:
        # which kinematics have exceeded the allowed deviations
        pos_is = np.where(pos_exceeded==True)[0]
        vel_is = np.where(vel_exceeded==True)[0]
        pos_labels, vel_labels = self.refs.get_labels_by_model_index(pos_is, vel_is)

        DEBUG_ET = False
        if DEBUG_ET and (pos_is.any() or vel_is.any()):
            print()
            for i_pos, pos_label in zip(pos_is, pos_labels):
                print(f"{pos_label} \t exceeded the allowed deviation ({int(100*max_dev_pos)}% "
                      f"of the max range of {pos_ranges[i_pos]}) after {self.refs.ep_dur} steps:"
                      f"{delta_pos[i_pos]}")

            print()
            for i_vel, vel_label in zip(vel_is, vel_labels):
                print(f"{vel_label} \t exceeded the allowed deviation ({int(100*max_dev_vel)}% "
                      f"of the max range of {vel_ranges[i_vel]}) after {self.refs.ep_dur} steps:"
                      f"{delta_vel[i_vel]}")

        return pos_exceeded.any() or vel_exceeded.any()

    # ----------------------------
    # Methods we override:
    # ----------------------------

    def close(self):
        # overwritten to set RSI init flag to False
        global _rsinitialized
        _rsinitialized = False
        # calls MujocoEnv.close()
        super().close()

    # ----------------------------
    # Methods to override:
    # ----------------------------

    def _get_COM_indices(self):
        """
        Needed to distinguish between joint and COM kinematics.

        Returns a list of indices pointing at COM joint position/index
        in the considered robot model, e.g. [0,1,2]
        """
        raise NotImplementedError

    def _get_not_actuated_joint_indices(self):
        """
        Needed for playing back reference trajectories
        by using position servos in the actuated joints.

        @returns a list of indices specifying indices of
        joints in the considered robot model that are not actuated.
        Example: return [0,1,2,6,7]
        """
        raise NotImplementedError