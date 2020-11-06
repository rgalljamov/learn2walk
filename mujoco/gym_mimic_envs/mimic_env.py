'''
Interface for environments using reference trajectories.
'''
import gym, mujoco_py
import numpy as np
from scripts.common import config as cfg
from scripts.common.utils import log, is_remote, \
    exponential_running_smoothing as smooth, resetExponentialRunningSmoothing as reset_smooth
from scripts.mocap.ref_trajecs import ReferenceTrajectories as RefTrajecs

# double max deltas for better perturbation recovery. To keep balance,
# the agent might require to output angles that are not reachable and saturate the motors.
SCALE_MAX_VELS = 2

# Workaround: MujocoEnv calls step() before calling reset()
# Then, RSI is not executed yet and ET gets triggered during step()
_rsinitialized = False

# flag if ref trajectories are played back
_play_ref_trajecs = False

# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_viewer_at_first_step = True and not is_remote()

# monitor episode and training duration
step_count = 0
ep_dur = 0

class MimicEnv:

    def __init__(self: gym.Env, ref_trajecs:RefTrajecs):
        '''@param: self: gym environment implementing the MimicEnv interface.'''
        self.refs = ref_trajecs
        # adjust simulation properties (to the frequency of the ref trajec)
        self._sim_freq, self._frame_skip = self.get_sim_freq_and_frameskip()
        self.model.opt.timestep = 1 / self._sim_freq
        self.control_freq = self._sim_freq / self._frame_skip
        self.refs.set_sampling_frequency(self.control_freq)
        # names of all robot kinematics
        self.kinem_labels = self.refs.get_kinematics_labels()
        # keep the body in the air for testing purposes
        self._FLY = False or cfg.is_mod(cfg.MOD_FLY)
        self._evaluation_on = False
        if cfg.is_mod(cfg.MOD_MAX_TORQUE):
            self.model.actuator_forcerange[:,:] = cfg.TORQUE_RANGES
        # for ET based on mocap distributions
        self.left_step_distrib, self.right_step_distrib = None, None
        # control desired walking speed
        self.SPEED_CONTROL = False
        # load kinematic stds for deviation normalization and better reward signal
        self.kinem_stds = np.load(cfg.abs_project_path +
                                  'assets/ref_trajecs/distributions/3d_mocap_stds_const_speed_400hz.npy')
        # self.kinem_stds = self.kinem_stds[self.refs.qpos_is + self.refs.qvel_is]
        # track different reward components
        self.pos_rew, self.vel_rew, self.com_rew = 0,0,0
        # track running mean of the return and use it for ET reward
        self.ep_rews = []
        self.mean_epret_smoothed = 0
        # dense grnd contact info
        self.last_contact_left = 0
        self.last_contact_right = 0
        self.left_contacts = []
        self.right_contacts = []

    def step(self, a):
        """
        Returns
        -------
        True if MimicEnv was already instantiated.
        Workaround (see doc string for _rsinitialized)
        """
        global _rsinitialized

        if not _rsinitialized:
            obs = self._get_obs()
            return obs, -3.33, False, {}

        try:
            if self.refs is None: pass
        except:
            # log("MimicEnv.step() called before refs were initialized!")
            _rsinitialized = False
            obs = self._get_obs()
            return obs, -3.33, False, {}

        self.joint_pow_sum_normed = self.get_joint_power_sum_normed()
        self.refs.next()

        # pause sim on startup to be able to change rendering speed, camera perspective etc.
        global pause_viewer_at_first_step
        if pause_viewer_at_first_step:
            self._get_viewer('human')._paused = True
            pause_viewer_at_first_step = False

        DEBUG = False

        global step_count, ep_dur
        step_count += 1
        ep_dur += 1

        qpos_before = np.copy(self.sim.data.qpos)
        qvel_before = np.copy(self.sim.data.qvel)

        posbefore = qpos_before[0]

        # hold the agent in the air
        if self._FLY:
            # get current joint angles and velocities
            qpos_set = np.copy(qpos_before)
            qvel_set = np.copy(qvel_before)
            # fix COM position, trunk rotation and corresponding velocities
            qpos_set[[0, 1, 2]] = [0, 1.2, 0]
            qvel_set[[0, 1, 2, ]] = [0, 0, 0]
            self.set_joint_kinematics_in_sim(qpos_set, qvel_set)

        # save qpos of actuated joints for reward calculation
        qpos_act_before_step = self.get_qpos(True, True)

        # policy outputs qpos deltas
        if cfg.is_mod(cfg.MOD_PI_OUT_DELTAS):
            if cfg.is_mod(cfg.MOD_NORM_ACTS):
                # rescale actions to actual bounds
                a *= self.get_max_qpos_deltas()
            a = qpos_act_before_step + a
        elif cfg.is_mod(cfg.MOD_NORM_ACTS):
            qpos_min, qpos_max = self.get_qpos_ranges()
            # qpos_min: [-0.8727 -0.7854  0.     -0.3491 -0.8727 -0.0873  0.     -0.3491]
            # qpos_max: [0.8727  0.0873   2.618  0.6981  0.8727  0.7854   2.618  0.6981]
            if cfg.is_mod(cfg.MOD_NORM_CONSIDER_SIGN):
                # todo: add mean actions. this way we shift the normal distribution to be around the mean...
                a_unnormalized = []
                for i, act in enumerate(a):
                    if act > 0:
                        a_unnormalized.append(qpos_max[i]*act)
                    else:
                        a_unnormalized.append(qpos_min[i]*np.abs(act))
                a = np.array(a_unnormalized)
            else:
                act_prct = (a + 1) / 2
                qpos_ranges = (qpos_max - qpos_min)
                a = qpos_min + act_prct * qpos_ranges

        if cfg.is_mod(cfg.MOD_TORQUE_DELTAS):
            last_action = np.copy(self.sim.data.actuator_force)
            deltas = a * cfg.trq_delta
            a = last_action + deltas

        if cfg.is_mod(cfg.MOD_MIRR_STEPS) and self.refs.is_step_left():
            a = self.mirror_acts(a)

        # execute simulation with desired action for multiple steps
        self.do_simulation(a, self._frame_skip)

        qpos_after = self.sim.data.qpos
        qvel_after = self.sim.data.qvel

        posafter, _, ang = qpos_after[0:3]
        com_z_pos = qpos_after[self._get_COM_indices()[-1]]

        qpos_delta0 = qpos_before[0] - qpos_after[0]
        qpos_delta = qpos_after - qpos_before

        alive_bonus = 1.0
        com_x_vel_finite_difs = (posafter - posbefore) / self.dt
        com_x_vel_qvel = self.sim.data.qvel[0]
        com_x_vel_delta = com_x_vel_finite_difs - com_x_vel_qvel
        qvel_findifs = (qpos_delta) / self.dt
        qvel_delta = qvel_after - qvel_findifs

        # get state observation after simulation step
        obs = self._get_obs()

        USE_DMM_REW = True and not cfg.do_run()
        if USE_DMM_REW:
            reward = self.get_imitation_reward(qpos_act_before_step, a)
            self.ep_rews.append(reward)
        else:
            reward = (com_x_vel_finite_difs)
            reward += alive_bonus
            reward -= 1e-3 * np.square(a).sum()

        USE_ET = False or cfg.is_mod(cfg.MOD_REF_STATS_ET)
        USE_REW_ET = True and not cfg.do_run() and not USE_ET
        walked_distance = qpos_after[0]
        is_ep_end = ep_dur >= cfg.ep_dur_max or walked_distance > cfg.max_distance + 0.01
        if self.is_evaluation_on():
            # is_ep_end = ep_dur >= cfg.ep_dur_max or walked_distance > 22.05
            done = com_z_pos < 0.5 or is_ep_end
        elif USE_ET:
            if cfg.is_mod(cfg.MOD_REF_STATS_ET):
                done = self.is_out_of_ref_distribution(obs)
            else:
                done = self.has_exceeded_allowed_deviations()
        elif USE_REW_ET:
            done, com_height_too_low, trunk_ang_exceeded, is_drunk, \
            rew_too_low = self.do_terminate_early(reward, rew_threshold=cfg.et_rew_thres)
            done = done or is_ep_end
        else:
            done = not (com_z_pos > 0.8 and com_z_pos < 2.0 and
                        ang > -1.0 and ang < 1.0)
            done = done or is_ep_end
        if DEBUG and done: print('Done')

        # punish episode termination
        if done:
            # calculated a running mean of the ep_return
            self.mean_epret_smoothed = smooth('mimic_env_epret', np.sum(self.ep_rews), 0.5)
            self.ep_rews = []

            # don't punish if episode just reached max length
            if not is_ep_end:
                # punish only falling hard
                if not self.is_evaluation_on():
                    et_rew = -1 * self.mean_epret_smoothed
                    if rew_too_low:
                        reward = -1
                    elif com_height_too_low or trunk_ang_exceeded:
                        reward = et_rew
                    else:
                        reward = 0.5 * et_rew
            else:
                # reward = cfg.ep_end_reward

                # estimate mean state value / mean action return
                mean_step_rew = self.mean_epret_smoothed / ep_dur
                act_ret_est = np.sum(mean_step_rew * np.power(cfg.gamma, np.arange(ep_dur)))
                reward = act_ret_est
                # print(f'Successfully reached the end of the episode '
                #       f'after {ep_dur} steps and got reward of ', reward)
            ep_dur = 0

        if cfg.is_mod(cfg.MOD_REW_DELTA):
            rew_delta = reward - self.last_reward
            # print(f'cur. reward: \t {reward}\n'
            #       f'last reward: \t {self.last_reward}\n'
            #       f'reward delta:\t {rew_delta}\n')
            self.last_reward = reward
            reward = reward + cfg.rew_delta_scale * rew_delta
            if reward < 0: reward = 0

        if walked_distance > cfg.alive_min_dist and not done:
            reward += cfg.alive_bonus
            # alive_bonus = 1 * cfg.rew_scale * step_count / (cfg.mio_steps * 1e6)
            # reward += alive_bonus

        return obs, reward, done, {}

    def get_sim_freq_and_frameskip(self):
        """
        What is the frequency the simulation is running at
        and how many frames should be skipped during step(action)?
        """
        SIM_FREQ = 1000
        return SIM_FREQ, int(SIM_FREQ/cfg.CTRL_FREQ)

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

    def _exclude_joints(self, qvals, exclude_com, exclude_not_actuated_joints):
        if exclude_not_actuated_joints:
            qvals = self._remove_by_indices(qvals, self._get_not_actuated_joint_indices())
        elif exclude_com:
            qvals = self._remove_by_indices(qvals, self._get_COM_indices())
        return qvals

    def get_qpos(self, exclude_com=False, exclude_not_actuated_joints=False):
        qpos = np.copy(self.sim.data.qpos)
        return self._exclude_joints(qpos, exclude_com, exclude_not_actuated_joints)

    def get_qvel(self, exclude_com=False, exclude_not_actuated_joints=False):
        qvel = np.copy(self.sim.data.qvel)
        return self._exclude_joints(qvel, exclude_com, exclude_not_actuated_joints)

    def get_ref_qpos(self, exclude_com=False, exclude_not_actuated_joints=False):
        qpos = self.refs.get_qpos()
        return self._exclude_joints(qpos, exclude_com, exclude_not_actuated_joints)

    def get_ref_qvel(self, exclude_com=False, exclude_not_actuated_joints=False):
        qvel = self.refs.get_qvel()
        return self._exclude_joints(qvel, exclude_com, exclude_not_actuated_joints)

    def activate_evaluation(self):
        self._evaluation_on = True

    def is_evaluation_on(self):
        return self._evaluation_on

    def do_fly(self, fly=True):
        self._FLY = fly

    def get_actuator_torques(self, abs_mean=False):
        tors = np.copy(self.sim.data.actuator_force)
        # if ctrl_force is [-1:1] and gear = 300
        if np.max(tors) <= 1:
            tors *= 300
        return np.mean(np.abs(tors)) if abs_mean else tors

    def get_force_ranges(self):
        return np.copy(self.model.actuator_forcerange)

    def get_qpos_ranges(self):
        ctrl_ranges = np.copy(self.model.actuator_ctrlrange)
        low = ctrl_ranges[:, 0]
        high = ctrl_ranges[:, 1]
        return low, high

    def get_max_qpos_deltas(self):
        """Returns the scalars needed to rescale the normalized actions from the agent."""
        # get max allowed deviations in actuated joints
        max_vels = self._get_max_actuator_velocities()
        # double max deltas for better perturbation recovery
        # to keep balance the agent might require to output
        # angles that are not reachable to saturate the motors
        max_qpos_deltas = SCALE_MAX_VELS * max_vels / self.control_freq
        return max_qpos_deltas

    def playback_ref_trajectories(self, timesteps=2000, pd_pos_control=False):
        global _play_ref_trajecs
        _play_ref_trajecs = True

        from gym_mimic_envs.monitor import Monitor
        env = Monitor(self)

        env.reset()
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
                obs, reward, done, _ = env.step(des_qpos)
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

    def set_action_space_deltas(self):
        if cfg.is_mod(cfg.MOD_NORM_ACTS):
            ones = np.ones_like(self.action_space.high)
            self.action_space.high = ones
            self.action_space.low = -1*ones
            return
        # get max allowed deviations in actuated joints
        max_vels = self._get_max_actuator_velocities()
        max_qpos_deltas = max_vels/self.control_freq
        self.action_space.high = SCALE_MAX_VELS * max_qpos_deltas
        self.action_space.low = SCALE_MAX_VELS * -max_qpos_deltas
        if cfg.DEBUG:
            log(f'Changed action space to scaled ({SCALE_MAX_VELS}x) deltas:'
                f'\nhigh: {self.action_space.high}, low: {self.action_space.low}')

    def set_joint_kinematics_in_sim(self, qpos=None, qvel=None):
        old_state = self.sim.get_state()
        if qpos is None:
            qpos, qvel = self.refs.get_ref_kinmeatics()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)

    def activate_speed_control(self, desired_walking_speeds):
        self.SPEED_CONTROL = True
        self.desired_walking_speeds = desired_walking_speeds
        self.i_speed = 0

    def _get_obs(self):
        global _rsinitialized
        qpos, qvel = self.get_joint_kinematics()
        # remove COM x position as the action should be independent of it
        qpos = qpos[1:]
        if _rsinitialized and not cfg.approach == cfg.AP_RUN:

            if self.SPEED_CONTROL:
                self.desired_walking_speed = self.desired_walking_speeds[self.i_speed]
                self.i_speed += 1
                if self.i_speed >= len(self.desired_walking_speeds): self.i_speed = 0
            else:
                self.desired_walking_speed = self.refs.get_step_velocity()

            phase = self.refs.get_phase_variable()
        else:
            self.desired_walking_speed = -3.33
            phase = 0
        if cfg.approach == cfg.AP_RUN:
            obs = np.concatenate([qpos, qvel]).ravel()
        else:
            obs = np.concatenate([np.array([phase, self.desired_walking_speed]), qpos, qvel]).ravel()

        if cfg.is_mod(cfg.MOD_GROUND_CONTACT):
            if _rsinitialized and cfg.is_mod(cfg.MOD_GROUND_CONTACT_DENSE):

                label_left = 'grd_contact_left'
                label_right = 'grd_contact_right'
                has_contact_left, has_contact_right = np.array(self.has_ground_contact()).astype(np.float)

                # avoid both feet having no contact with the ground:
                if has_contact_right == 0 and has_contact_left == 0:
                    has_contact_right = self.last_contact_right
                    has_contact_left = self.last_contact_left

                if self.last_contact_left != has_contact_left:
                    reset_smooth(label_left, self.last_contact_left)
                if self.last_contact_right != has_contact_right:
                    reset_smooth(label_right, self.last_contact_right)
                self.last_contact_left = has_contact_left
                self.last_contact_right = has_contact_right

                has_contact_left = smooth(label_left, has_contact_left, 0.0025)
                has_contact_right = smooth(label_right, has_contact_right, 0.0025)
                # self.left_contacts.append(has_contact_left)
                # self.right_contacts.append(has_contact_right)
                #
                # if len(self.left_contacts) >= 1600:
                #     from matplotlib import pyplot as plt
                #     plt.subplot(1,2,1)
                #     plt.plot(self.left_contacts)
                #     plt.subplot(1,2,2)
                #     plt.plot(self.right_contacts)
                #     plt.show()
                #     exit(33)
                has_contact = np.array([has_contact_left, has_contact_right]).astype(np.float)
                obs = np.concatenate([has_contact, obs]).ravel()
            else:
                has_contact = np.array(self.has_ground_contact()).astype(np.float)
                obs = np.concatenate([has_contact, obs]).ravel()

        if _rsinitialized and cfg.is_mod(cfg.MOD_MIRR_STEPS) and self.refs.is_step_left():
            assert not cfg.is_mod(cfg.MOD_GROUND_CONTACT)
            obs = self.mirror_obs(obs)
        # round obs
        # obs = np.round(obs, decimals=3)
        return obs


    def mirror_obs(self, obs):
        is3d = '3d' in cfg.env_name or '3pd' in cfg.env_name
        if is3d:
            # 3D Walker obs indices:
            #           0: phase, 1: des_vel, 2: com_y, 3: com_z,
            #           4: trunk_rot_x, 5: trunk_rot_y, 6: trunk_rot_z,
            #           7: hip_ang_r_sag, 8: hip_ang_r_front, 9: knee_ang_r, 10: ankle_ang_r,
            #           11: hip_ang_l_sag, 12: hip_ang_l_front 13: knee_ang_l, 14: ankle_ang_l,
            #           15: com_x_vel, 16: com_y_vel, 17:com_z_vel,
            #           18: trunk_x_ang_vel, 19: trunk_y_ang_vel, 20: trunk_z_ang_vel,
            #           21: hip_sag_vel_r, 22: hip_front_vel_r, 23: knee_vel_r, 24: ankle_vel_r,
            #           25: hip_sag_vel_l, 26: hip_front_vel_l, 27: knee_vel_l, 28: ankle_vel_l
            mirred_obs_indices = [0, 1, 2, 3,
                                  4, 5, 6,
                                  11, 12, 13, 14,
                                  7, 8, 9, 10,
                                  15, 16, 17,
                                  18, 19, 20,
                                  25, 26, 27, 28,
                                  21, 22, 23, 24]
            # some observations and actions retain the same absolute value but change the sign
            negate_obs_indices = [2, 4, 6, 8, 12, 16, 18, 20, 22, 26]
        else:
            # 2D Walker obs indices:
            #           0: phase, 1: des_vel, 2: com_z, 3: trunk_rot,
            #           4: hip_ang_r, 5: knee_ang_r, 6: ankle_ang_r,
            #           7: hip_ang_l, 8: knee_ang_l, 9: ankle_ang_l,
            #           10: com_x_vel, 11:com_z_vel, 12: trunk_ang_vel,
            #           13: hip_vel_r, 14: knee_vel_r, 15: ankle_vel_r,
            #           16: hip_vel_l, 17: knee_vel_l, 18: ankle_vel_l
            mirred_obs_indices = [0, 1, 2, 3, 7, 8, 9, 4, 5, 6,
                                  10, 11, 12, 16, 17, 18, 13, 14, 15]

        obs_mirred = obs[mirred_obs_indices]

        if is3d:
            obs_mirred[negate_obs_indices] *= -1

        return obs_mirred


    def mirror_acts(self, acts):
        is3d = '3d' in cfg.env_name or '3pd' in cfg.env_name
        if is3d:
            mirred_acts_indices = [4, 5, 6, 7, 0, 1, 2, 3]
            # some observations and actions retain the same absolute value but change the sign
            negate_act_indices = [1, 5]
        else:
            mirred_acts_indices = [3, 4, 5, 0, 1, 2]

        try:
            acts_mirred = acts[mirred_acts_indices]
            if is3d:
                acts_mirred[negate_act_indices] *= -1
        except Exception as ex:
            log('WARNING! Could not mirror actions',
                [f'Exception: {ex}', f'Actions: {acts}'])
            acts_mirred = acts


        return acts_mirred


    def reset_model(self):
        '''WARNING: This method seems to be specific to MujocoEnv.
           Other gym environments just use reset().'''
        self.joint_pow_sum_normed = 0
        qpos, qvel = self.get_init_state(not self.is_evaluation_on() and not self.SPEED_CONTROL)

        ### avoid huge joint toqrues from PD servos after RSI
        # Explanation: on reset, ctrl is set to all zeros.
        # When we set a desired state during RSI, we suddenly change the current state.
        # Without also changing the target angles, there will be a huge difference
        # between target and current angles and PD Servos will kick in with high torques
        # todo: not sure we need it. The agent should have the chance to choose a first action
        # todo: ... and overwrite the zero ctrl!
        # if not cfg.is_torque_model:
        #     self.data.ctrl[:] = self._remove_by_indices(qpos, self._get_not_actuated_joint_indices())

        self.set_state(qpos, qvel)
        # check reward function
        rew = self.get_imitation_reward()
        self.last_reward = rew
        assert (rew > 0.95 * cfg.rew_scale) if not self._FLY else 0.5, \
            f"Reward should be around 1 after RSI, but was {rew}!"
        # assert not self.has_exceeded_allowed_deviations()
        # set the reference trajectories to the next state,
        # otherwise the first step after initialization has always zero error
        self.refs.next()
        # print(self.refs._i_step, self.refs._pos)
        obs = self._get_obs()
        # print(f'init obs: {obs[:8]}')
        return obs


    def get_init_state(self, random=True):
        ''' Random State Initialization:
            @returns: qpos and qvel of a random step at a random position'''
        global _rsinitialized
        _rsinitialized = True
        return self.refs.get_random_init_state() if random \
            else self.refs.get_deterministic_init_state()


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

        if cfg.is_mod(cfg.MOD_IMPROVE_REW):
            difs = np.abs(qpos - ref_pos)
            # normalize deviations to treat all joints the same
            # use two standard deviations of refs for normalization
            ref_pos_stds = 2 * self.kinem_stds[self.refs.qpos_is[2:]]
            difs_normed = difs / ref_pos_stds
            difs_mean = np.mean(difs_normed)
            exp_rew = np.exp(-2.5 * difs_mean)

            if cfg.is_mod(cfg.MOD_LIN_REW):
                lin_rew = 1 - difs_mean * (0.5 if cfg.is_mod('half_slope') else 1)
                return lin_rew
            else:
                return exp_rew


        dif = qpos - ref_pos
        dif_sqrd = np.square(dif)
        sum = np.sum(dif_sqrd)
        pose_rew = np.exp(-3 * sum)
        return pose_rew

    def get_vel_reward(self):
        _, qvel = self.get_joint_kinematics(exclude_com=True)
        _, ref_vel = self.get_ref_kinematics(exclude_com=True)
        if self._FLY:
            # set trunk angular velocity to zero
            ref_vel[0] = 0
        if cfg.is_mod(cfg.MOD_IMPROVE_REW):
            difs = np.abs(qvel - ref_vel)
            # normalize deviations to treat all joints the same
            # use two standard deviations or refs for normalization
            ref_vel_stds = 2 * self.kinem_stds[self.refs.qvel_is[2:]]
            dif_normed = difs / ref_vel_stds
            dif_mean = np.mean(dif_normed)
            if cfg.is_mod(cfg.MOD_LIN_REW):
                vel_rew = 1 - dif_mean * (0.5 if cfg.is_mod('half_slope') else 1)
            else:
                vel_rew = np.exp(-2.5*dif_mean)
        else:
            difs = qvel - ref_vel
            dif_sqrd = np.square(difs)
            dif_sum = np.sum(dif_sqrd)
            vel_rew = np.exp(-0.05 * dif_sum)
        return vel_rew

    def get_com_reward(self):
        qpos, qvel = self.get_joint_kinematics()
        ref_pos, ref_vel = self.get_ref_kinematics()
        com_is = self._get_COM_indices()

        if cfg.is_mod(cfg.MOD_IMPROVE_REW):
            USE_COM_XVEL = cfg.is_mod(cfg.MOD_COM_X_VEL)
            if USE_COM_XVEL:
                com_pos = np.array([qpos[1], qvel[0]])
                com_ref = np.array([ref_pos[1], ref_vel[0]])
                # normalize the differences with 2*stds
                max_deviats = [2 * self.kinem_stds[self.refs.qpos_is[1]],
                               2 * self.kinem_stds[self.refs.qvel_is[0]]]
            else:
                com_pos, com_ref = qpos[com_is], ref_pos[com_is]
                # normalize the differences with 2stds
                inds = np.array(self.refs.qpos_is)[com_is]
                max_deviats = 2 * self.kinem_stds[inds]

            dif = np.abs(com_pos - com_ref)
            dif_normed = dif / max_deviats
            # dif_sqrd = np.square(dif_normed)
            mean_dif_normed = np.mean(dif_normed)
            exp_rew = np.exp(-2.5 * mean_dif_normed)
            if cfg.is_mod(cfg.MOD_LIN_REW):
                return 1 - mean_dif_normed * (0.5 if cfg.is_mod('half_slope') else 1)
            return exp_rew

        com_pos, com_ref = qpos[com_is], ref_pos[com_is]
        dif = com_pos - com_ref
        dif_sqrd = np.square(dif)
        sum = np.sum(dif_sqrd)
        com_rew = np.exp(-16 * sum)
        return com_rew

    def get_energy_reward(self):
        """
        Reward based on percentage of max possible joint power.
        """
        pow_prct = self.joint_pow_sum_normed
        energy_rew = 1 - pow_prct
        assert energy_rew <= 1, \
            f'Energy Reward should be between 0 and 1 but was {energy_rew}'
        return energy_rew

    def get_joint_power_sum_normed(self):
        torques = np.abs(self.get_actuator_torques())
        max_tors = self.get_force_ranges().max(axis=1)
        # log(f'Max Torques: {max_tors}')
        qvels = np.abs(self.get_qvel(exclude_not_actuated_joints=True))
        max_vels = self._get_max_actuator_velocities()
        assert torques.shape == max_tors.shape
        assert qvels.shape == max_vels.shape
        pows = np.multiply(torques, qvels)
        max_pows = np.multiply(max_tors, max_vels)
        abs_pows = np.abs(pows)
        abs_max_pows = np.abs(max_pows)
        sum_pows = np.sum(abs_pows)
        sum_max_pows = np.sum(abs_max_pows)
        pow_normed = sum_pows / sum_max_pows
        return pow_normed

    def get_angle_deltas_reward(self, qpos_act, action):
        """
        Goal: Prevent the agent from outputting unrealistic target angles.
        Example of unrealistic angles: current is 45°, desired -45°.
        Realistic target angles (actions) are such close to the previous qpos.
        """
        # ignore zero qpos during environment reset
        if np.count_nonzero(qpos_act) == 0 : return 1

        # get max allowed deviations in actuated joints
        max_vels = self._get_max_actuator_velocities()
        # double max deltas for better perturbation recovery
        # to keep balance the agent might require to output
        # angles that are not reachable to saturate the motors
        max_qpos_deltas = SCALE_MAX_VELS * max_vels / self.control_freq
        # get angle deltas
        qpos_deltas_abs = np.abs(action - qpos_act)
        qpos_deltas_sum = np.sum(qpos_deltas_abs)
        # determine if max deviations were exceeded
        qpos_delta_too_high = qpos_deltas_abs > (max_qpos_deltas + 0.01)
        if qpos_delta_too_high.any(): return 0.75
        else: return 1

        return np.exp(-0.1 * qpos_deltas_sum)

    def _remove_by_indices(self, list, indices):
        """
        Removes specified indices from the passed list and returns it.
        """
        new_list = [item for i, item in enumerate(list) if i not in indices]
        return np.array(new_list)

    def get_imitation_reward(self, qpos_act=None, action=None):
        """
        :param qpos_act: qpos of actuated joints before step() exectuion.
        :param action: target angles for PD position controller
        """
        global _rsinitialized
        if not _rsinitialized:
            return -3.33

        # get rew weights from rew_weights_string
        weights = [float(digit)/10 for digit in cfg.rew_weights]

        w_pos, w_vel, w_com, w_pow = weights
        pos_rew = self.get_pose_reward()
        vel_rew = self.get_vel_reward()
        com_rew = self.get_com_reward()
        pow_rew = self.get_energy_reward() if w_pow != 0 else 0

        self.pos_rew, self.vel_rew, self.com_rew = pos_rew, vel_rew, com_rew

        if cfg.is_mod(cfg.MOD_REW_MULT):
            imit_rew = np.sqrt(pos_rew) * np.sqrt(com_rew) # * vel_rew**w_vel
        else:
            imit_rew = w_pos * pos_rew + w_vel * vel_rew + w_com * com_rew + w_pow * pow_rew

        if cfg.is_mod(cfg.MOD_PUNISH_UNREAL_TARGET_ANGS):
            # punish unrealistic joint target angles
            delta_rew = self.get_angle_deltas_reward(qpos_act, action)
            imit_rew *= delta_rew

        return imit_rew * cfg.rew_scale

    def do_terminate_early(self, rew, rew_threshold = 0.05):
        """
        Early Termination based on reward, falling and episode duration
        """
        if (not _rsinitialized) or _play_ref_trajecs:
            # ET only works after RSI was executed
            return False

        qpos = self.get_qpos()
        ref_qpos = self.refs.get_qpos()
        com_indices = self._get_COM_indices()
        trunk_ang_indices = self._get_trunk_joint_indices()

        com_height = qpos[com_indices[-1]]
        com_y_pos = qpos[com_indices[1]]
        trunk_angs = qpos[trunk_ang_indices]
        ref_trunk_angs = ref_qpos[trunk_ang_indices]

        # calculate if trunk saggital angle is out of allowed range
        max_pos_sag = 0.3
        max_neg_sag = -0.05
        is2d = len(trunk_ang_indices) == 1
        if is2d:
            trunk_ang_saggit = trunk_angs # is the saggital ang
            sag_dev = np.abs(trunk_ang_saggit - ref_trunk_angs)
            trunk_ang_sag_exceeded = trunk_ang_saggit > max_pos_sag or trunk_ang_saggit < max_neg_sag
            trunk_ang_exceeded = trunk_ang_sag_exceeded
        else:
            max_front_dev = 0.2
            max_axial_dev = 0.5 # should be much smaller but actually doesn't really hurt performance
            trunk_ang_front, trunk_ang_saggit, trunk_ang_axial = trunk_angs
            front_dev, sag_dev, ax_dev = np.abs(trunk_angs - ref_trunk_angs)
            trunk_ang_front_exceeded = front_dev > max_front_dev
            trunk_ang_axial_exceeded = ax_dev > max_axial_dev
            trunk_ang_sag_exceeded = trunk_ang_saggit > max_pos_sag or trunk_ang_saggit < max_neg_sag
            trunk_ang_exceeded = trunk_ang_sag_exceeded or trunk_ang_front_exceeded \
                                 or trunk_ang_axial_exceeded

        # check if agent has deviated too much in the y direction
        is_drunk = np.abs(com_y_pos) > 0.15

        # is com height too low (e.g. walker felt down)?
        min_com_height = 0.95
        com_height_too_low = com_height < min_com_height
        if self._FLY: com_height_too_low = False

        rew_too_low = False # rew < rew_threshold
        terminate_early = com_height_too_low or trunk_ang_exceeded or is_drunk or rew_too_low
        if False: #(terminate_early and np.random.randint(low=1, high=500) == 7) \
                # or np.random.randint(low=1, high=77700) == 777:
             log(("" if terminate_early else "NO ") + "Early Termination",
                [f'COM Z too low:   \t{com_height_too_low} \t {com_height}',
                 f'COM Y too high:  \t{is_drunk} \t {com_y_pos}',
                 f'Trunk Angles:    \t{trunk_ang_exceeded} \t '
                    f'{sag_dev if is2d else np.round([front_dev, sag_dev, ax_dev], 3)}',
                 # f'Reward too low:  \t{rew_too_low} \t {rew}',
                 f'Reward was:      \t {"    "} \t {rew}'
                 ])
        return terminate_early, com_height_too_low, trunk_ang_exceeded, is_drunk, rew_too_low


    def is_out_of_ref_distribution(self, state, scale_pos_std=2, scale_vel_std=10):
        """
        Early Termination based on reference trajectory distributions:
           @returns: True if qpos and qvel are out of the reference trajectory distributions.
           @params: indicate how many stds deviation are allowed before being out of distribution.
        """

        if not cfg.is_mod(cfg.MOD_REFS_RAMP):
            # load trajectory distributions if not done already
            if self.left_step_distrib is None:
                npz = np.load(cfg.abs_project_path +
                                   'assets/ref_trajecs/distributions/2d_distributions_const_speed_400hz.npz')
                self.left_step_distrib = [npz['means_left'], npz['stds_left']]
                self.right_step_distrib = [npz['means_right'], npz['stds_right']]
                self.step_len = min(self.left_step_distrib[0].shape[1], self.right_step_distrib[0].shape[1])

            # left and right step distributions are different
            step_dist = self.left_step_distrib if self.refs.is_step_left() else self.right_step_distrib

            # get current mean on the mocap distribution, exlude com_x_pos
            pos = min(self.refs._pos, self.step_len-1) # todo: think of a better handling of longer then usual steps
            mean_state = step_dist[0][1:, pos]
            # check distance of current state to the distribution mean
            deviation = np.abs(mean_state - state[2:])
            # terminate if distance is too big, ignore com x pos
            std_state = 3*step_dist[1][1:, pos]
            is_out = deviation > std_state
            et = ( any((is_out[2:8])) or any((is_out[11:17])) ) if cfg.is_mod(cfg.MOD_FLY) \
                else any((is_out[:9]))
            et = any((is_out[:9]))
            # if et:
            #     print(self.refs.ep_dur)
            return et


        else: return NotImplementedError('Refs Distributions were so far only calculated '
                                         'for the constant speed trajectories!')


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

        Caution: Do not include trunk joints here.
        """
        raise NotImplementedError

    def _get_trunk_joint_indices(self):
        """ Return the index of the trunk euler rotation in the saggital plane. """
        raise NotImplementedError

    def _get_not_actuated_joint_indices(self):
        """
        Needed for playing back reference trajectories
        by using position servos in the actuated joints.

        @returns a list of indices specifying indices of
        joints in the considered robot model that are not actuated.
        Example: return [0,1,2]
        """
        raise NotImplementedError

    def _get_max_actuator_velocities(self):
        """
        Needed to calculate the max mechanical power for the energy efficiency reward.
        :returns: A numpy array containing the maximum absolute velocity peaks
                  of each actuated joint. E.g. np.array([6, 12, 12, 6, 12, 12])
        """
        raise NotImplementedError


    def has_ground_contact(self):
        """
        :returns: two booleans indicating ground contact of left and right foot.
        Example: [True, False] means left foot has ground contact, [True, True] indicates double stance.
        """
        raise NotImplementedError
