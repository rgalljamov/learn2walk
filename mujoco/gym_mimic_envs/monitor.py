import gym
import numpy as np
import seaborn as sns
from scripts.common.utils import config_pyplot, is_remote, exponential_running_smoothing as smooth
from gym_mimic_envs.mimic_env import MimicEnv
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# length of the buffer containing sim and ref trajecs for comparison
_trajec_buffer_length = 2000

PLOT_REF_DISTRIB =  False

class Monitor(gym.Wrapper):

    def __init__(self, env: MimicEnv):
        try:
            env_type = type(env)
            env_is_mimic_env = isinstance(env, MimicEnv)
            env_is_subproc = isinstance(env.venv, SubprocVecEnv)
            env_is_normalize = isinstance(env, VecNormalize)
            env_is_dummy = isinstance(env, DummyVecEnv)
        except:
            pass
        if isinstance(env, MimicEnv):
            self.env = env
        elif isinstance(env, DummyVecEnv):
            self.env = env.envs[0]

        super(Monitor, self).__init__(self.env)

        self.num_dofs = len(self.env.kinem_labels)
        # self.num_dofs = self.env.observation_space.high.size
        self.num_actions = self.env.action_space.high.size
        # do we want to control walking speed
        self.SPEED_CONTROL = False

        self.setup_containers()

        self.plt = config_pyplot(fullscreen=True, font_size=12,
                                 tick_size=12, legend_fontsize=16)

    def activate_speed_control(self, speeds):
        self.SPEED_CONTROL = True
        desired_walking_speeds = np.concatenate(
            [np.linspace(speeds[0], speeds[1], int(_trajec_buffer_length/6)),
            np.linspace(speeds[1], speeds[0], int(_trajec_buffer_length/6))])
        desired_walking_speeds = np.concatenate((desired_walking_speeds, desired_walking_speeds, desired_walking_speeds))
        self.env.activate_speed_control(desired_walking_speeds)

    def setup_containers(self):
        self.ep_len = 0
        self.reward = 0
        self.ep_len_smoothed = 0
        self.rewards = []
        # mean reward per step, calculated at each episode end
        self.mean_reward_smoothed = 0
        self.returns = []
        self.ep_ret_smoothed = 0
        self.ep_lens = []
        self.grfs_left = []
        self.grfs_right = []
        self.moved_distance_smooth = 0

        # monitor energy efficiency
        self.ep_torques_abs = []
        self.mean_abs_torque_smoothed = 0
        self.median_abs_torque_smoothed = 0
        self.ep_joint_pow_sum_normed = []
        self.mean_ep_joint_pow_sum_normed_smoothed = 0

        # monitor sim and ref trajecs for comparison (sim/ref, kinem_indices, timesteps)
        # 3 and 4 in first dimension are for mean and std of ref trajec distribution
        self.trajecs_buffer = np.zeros((4, self.num_dofs, _trajec_buffer_length))
        # monitor episode terminations
        self.dones_buf = np.zeros((_trajec_buffer_length,))
        # monitor the actions at actuated joints (PD target angles)
        self.action_buf = np.zeros((self.num_actions, _trajec_buffer_length))
        # monitor the joint torques
        self.torque_buf = np.zeros((self.num_actions, _trajec_buffer_length))
        # monitor desired walking speed
        self.speed_buf = np.zeros((_trajec_buffer_length,))

        self.left_step_distrib, self.right_step_distrib = None, None


    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        self.reward = reward
        self.ep_len += 1
        self.rewards.append(reward)
        self.ep_joint_pow_sum_normed.append(self.env.joint_pow_sum_normed)
        self.ep_torques_abs.append(self.env.get_actuator_torques(True))

        if done:
            ep_rewards = self.rewards[-self.ep_len:]
            mean_reward = np.mean(ep_rewards[:-1])
            self.mean_reward_smoothed = smooth('rew', mean_reward)

            ep_return = np.sum(ep_rewards)
            self.returns.append(ep_return)
            self.ep_ret_smoothed = smooth('ep_ret', ep_return, 0.25)

            self.ep_lens.append(self.ep_len)
            self.ep_len_smoothed = smooth('ep_len', self.ep_len, 0.75)
            self.ep_len = 0

            self.moved_distance_smooth = smooth('dist', self.env.data.qpos[0], 0.25)

            self.mean_abs_torque_smoothed = \
                smooth('mean_ep_tor', np.mean(self.ep_torques_abs), 0.75)
            self.median_abs_torque_smoothed = \
                smooth('med_ep_tor', np.median(self.ep_torques_abs), 0.75)
            self.ep_torques_abs = []

            self.mean_ep_joint_pow_sum_normed_smoothed = \
                smooth('joint_pow', np.mean(self.ep_joint_pow_sum_normed), 0.75)

        COMPARE_TRAJECS = True and not is_remote()
        if COMPARE_TRAJECS:

            # save sim and ref trajecs in a buffer for comparison
            sim_trajecs = self.env.get_joint_kinematics(concat=True)
            ref_trajecs = self.env.get_ref_kinematics(concat=True)
            # fifo approach, replace oldest entry with the newest one
            self.trajecs_buffer = np.roll(self.trajecs_buffer, -1, axis=2)
            self.trajecs_buffer[0, :, -1] = sim_trajecs
            self.trajecs_buffer[1, :, -1] = ref_trajecs

            if PLOT_REF_DISTRIB:
                # load trajectory distributions if not done already
                if self.left_step_distrib is None:
                    from scripts.common import config as cfg
                    npz = np.load(cfg.abs_project_path +
                                  'assets/ref_trajecs/distributions/const_speed_400hz.npz')
                    self.left_step_distrib = [npz['means_left'], npz['stds_left']]
                    self.right_step_distrib = [npz['means_right'], npz['stds_right']]
                    self.step_len = min(self.left_step_distrib[0].shape[1], self.right_step_distrib[0].shape[1])

                # left and right step distributions are different
                step_dist = self.left_step_distrib if self.refs.is_step_left() else self.right_step_distrib

                # get current mean on the mocap distribution, exlude com_x_pos
                pos = min(self.refs._pos, self.step_len - 1)
                mean_state = step_dist[0][:, pos]
                # terminate if distance is too big
                std_state = 3 * step_dist[1][:, pos]
                self.trajecs_buffer[2, :, -1] = mean_state
                self.trajecs_buffer[3, :, -1] = std_state

            # do the same with the dones
            self.dones_buf = np.roll(self.dones_buf, -1)
            self.dones_buf[-1] = done
            # and with the desired walking speed
            self.speed_buf = np.roll(self.speed_buf, -1)
            self.speed_buf[-1] = self.env.desired_walking_speed
            # save actions
            self.action_buf = np.roll(self.action_buf, -1, axis=1)
            self.action_buf[:, -1] = action
            # save joint toqrues
            self.torque_buf = np.roll(self.torque_buf, -1, axis=1)
            self.torque_buf[:, -1] = self.get_actuator_torques()

            # plot trajecs when the buffers are filled
            try: self.trajecs_recorded += 1
            except: self.trajecs_recorded = 1
            if self.trajecs_recorded % (1 * _trajec_buffer_length) == 0:
                self.compare_sim_ref_trajecs()

        return obs, reward, done, _


    def compare_sim_ref_trajecs(self):
        """
        Plot simulation and reference trajectories in a single figure
        to compare them.
        """
        plt = self.plt
        plt.rcParams.update({'figure.autolayout': False})
        sns.set_style("whitegrid", {'axes.edgecolor':'#ffffff00'})
        names = ['Simulation'] # line names (legend)
        second_y_axis_pos = 1.0

        if self.SPEED_CONTROL:
            # plt.rcParams.update({'axes.labelsize': 14})
            num_joints = 2
            rows, cols = 3, 1
            # only plot com x pos and velocity
            inds = [0, 9]
            self.trajecs_buffer = self.trajecs_buffer[:, inds, :]
            self.kinem_labels = self.kinem_labels[inds]
            y_labels = ['Moved Distance [m]', 'COM X Vel [m/s]']
        else:
            num_joints = len(self.kinem_labels)
            cols = 5
            rows = int((num_joints+1)/cols) + 1
        # plot sim trajecs
        trajecs = self.trajecs_buffer[0,:,:]
        # collect axes to reuse them for overlaying multiple plots
        axes = []
        # collect different lines to place the legend in a separate subplot
        lines = []
        for i_joint in range(num_joints):
            try: axes.append(plt.subplot(rows, cols, i_joint + 1, sharex=axes[i_joint-1]))
            except: axes.append(plt.subplot(rows, cols, i_joint + 1))
            trajec = trajecs[i_joint, :]
            line = plt.plot(trajec)
            # show episode ends
            plt.rcParams['lines.linewidth'] = 1
            plt.vlines(np.argwhere(self.dones_buf).flatten()+1,
                       np.min(trajec), np.max(trajec), colors='#cccccc', linestyles='dashed')
            plt.rcParams['lines.linewidth'] = 2
            if self.SPEED_CONTROL:
                plt.ylabel(y_labels[i_joint])
            else:
                plt.title(f'{i_joint}. ' + self.kinem_labels[i_joint])
        lines.append(line[0])

        # plot ref trajec distributions (mean + 2std)
        if PLOT_REF_DISTRIB:
            trajecs = self.trajecs_buffer[2,:,:]
            stds = self.trajecs_buffer[3,:,:]
            for i_joint in range(num_joints):
                trajec = trajecs[i_joint, :]
                std = stds[i_joint, :]
                line = axes[i_joint].plot(trajec)
                axes[i_joint].fill_between(range(len(trajec)), trajec+std, trajec-std,
                                           color='orange', alpha=0.5)
            lines.append(line[0])
            names.append('Reference Distribution\n(mean $\pm$ 2std)')

        PLOT_REFS = True
        if PLOT_REFS:
            trajecs = self.trajecs_buffer[1, :, :]
            for i_joint in range(num_joints):
                trajec = trajecs[i_joint, :]
                line = axes[i_joint].plot(trajec, color='red' if PLOT_REF_DISTRIB else 'orange')

            lines.append(line[0])
            names.append('Reference')

        def plot_actions(buffer, name, line_color='#777777'):
            with sns.axes_style("white", {"axes.edgecolor": '#ffffff00',
                                          "ytick.color":'#ffffff00'}):
                i_not_actuated = self.env._get_not_actuated_joint_indices()
                i_actuated = 0
                plt.rcParams['lines.linewidth'] = 1
                for i_joint in range(num_joints):
                    if i_joint in i_not_actuated:
                        continue
                    if i_actuated >= buffer.shape[0]:
                        break
                    act_plt = axes[i_joint].twinx()
                    act_plt.spines['right'].set_position(('axes', second_y_axis_pos))
                    line = act_plt.plot(buffer[i_actuated, :], line_color+'77')
                    act_plt.tick_params(axis='y', labelcolor=line_color)
                    i_actuated += 1
                plt.rcParams['lines.linewidth'] = 2
            lines.append(line[0])
            names.append(name)

        PLOT_TORQUES = False
        if PLOT_TORQUES:
            plot_actions(self.torque_buf/1000, "Joint Torque [kNm]")
            second_y_axis_pos = 1.12

        PLOT_ACTIONS = False
        if PLOT_ACTIONS:
            plot_actions(self.action_buf, 'PD Target', '#ff0000')

        if self.SPEED_CONTROL:
            plt.subplot(rows, cols, 3, sharex=axes[-1])
            plt.plot(self.speed_buf)
            plt.ylabel('Desired Walking Speed [m/s]')
            plt.xlabel('Simulation Timesteps []')
            axes[0].legend(lines, names)
        else:
            # plot the legend in a separate subplot
            with sns.axes_style("white", {"axes.edgecolor": 'white'}):
                legend_subplot = plt.subplot(rows, cols, num_joints + 2)
                legend_subplot.set_xticks([])
                legend_subplot.set_yticks([])
                legend_subplot.legend(lines, names, bbox_to_anchor=(
                    1.2 if PLOT_REF_DISTRIB else 1, 1.075 if PLOT_REF_DISTRIB else 1))

            # add rewards and returns
            rew_plot = plt.subplot(rows, cols, len(axes) + 1, sharex=axes[-1])
            rew_plot.plot(self.rewards[-_trajec_buffer_length:])
            rew_plot.set_ylim([-0.075, 1.025])
            # plot episode terminations
            plt.vlines(np.argwhere(self.dones_buf).flatten() + 1,
                       0, 1, colors='#cccccc')
            # plot episode returns
            ret_plot = rew_plot.twinx().twiny()
            ret_plot.plot(self.returns, '#77777777')
            ret_plot.tick_params(axis='y', labelcolor='#77777777')
            ret_plot.set_xticks([])

            plt.title('Rewards & Returns')

        # fix title overlapping when tight_layout is true
        plt.gcf().tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.55, hspace=0.5)
        PD_TUNING = False
        if PD_TUNING:
            rew_plot.set_xlim([-5, 250])
            dampings = self.env.model.dof_damping[3:].astype(int).tolist()
            kps = self.env.model.actuator_gainprm[:,0].astype(int).tolist()
            mean_rew = int(1000 * np.mean(self.rewards[-_trajec_buffer_length:]))
            plt.suptitle(f'PD Gains Tuning:   rew={mean_rew}    kp={kps}    kd={dampings}')
        else:
            plt.suptitle('Simulation and Reference Joint Kinematics over Time ' + \
                         '' if self.SPEED_CONTROL else '(Angles in [rad], Angular Velocities in [rad/s])')

        plt.show()
        if self.env.is_evaluation_on() or PD_TUNING:
            raise SystemExit('Planned exit after closing trajectory comparison plot.')