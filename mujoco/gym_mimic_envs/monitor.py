import gym
import numpy as np
import seaborn as sns
from scripts.common.utils import config_pyplot
from gym_mimic_envs.mimic_env import MimicEnv

# length of the buffer containing sim and ref trajecs for comparison
_trajec_buffer_length = 2000

class Monitor(gym.Wrapper):

    def __init__(self, env: MimicEnv):
        self.env = env
        super(Monitor, self).__init__(self.env)

        self.num_dofs = self.env.kinem_labels
        self.num_actions = self.env.action_space.high.size

        self.setup_containers()

        self.plt = config_pyplot(fullscreen=True, font_size=12,
                                 tick_size=12, legend_fontsize=16)


    def setup_containers(self):
        self.ep_len = 0
        self.rewards = []
        self.returns = []
        self.ep_lens = []
        self.grfs_left = []
        self.grfs_right = []

        # monitor sim and ref trajecs for comparison (sim/ref, kinem_indices, timesteps)
        self.trajecs_buffer = np.zeros((2, len(self.num_dofs), _trajec_buffer_length))
        # monitor episode terminations
        self.dones_buf = np.zeros((_trajec_buffer_length,))
        # monitor the actions at actuated joints (PD target angles)
        self.action_buf = np.zeros((self.num_actions, _trajec_buffer_length))
        # monitor the joint torques
        self.torque_buf = np.zeros((self.num_actions, _trajec_buffer_length))

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        self.ep_len += 1
        self.rewards.append(reward)

        if done:
            self.returns.append(np.sum(self.rewards[:-self.ep_len]))
            self.ep_lens.append(self.ep_len)
            self.ep_len = 0

        COMPARE_TRAJECS = True
        if COMPARE_TRAJECS:
            # save sim and ref trajecs in a buffer for comparison
            sim_trajecs = self.env.get_joint_kinematics(concat=True)
            ref_trajecs = self.env.get_ref_kinematics(concat=True)
            # fifo approach, replace oldest entry with the newest one
            self.trajecs_buffer = np.roll(self.trajecs_buffer, -1, axis=2)
            self.trajecs_buffer[0, :, -1] = sim_trajecs
            self.trajecs_buffer[1, :, -1] = ref_trajecs
            # do the same with the dones
            self.dones_buf = np.roll(self.dones_buf, -1)
            self.dones_buf[-1] = done
            # save actions
            self.action_buf = np.roll(self.action_buf, -1, axis=1)
            self.action_buf[:, -1] = action
            # save joint toqrues
            self.torque_buf = np.roll(self.torque_buf, -1, axis=1)
            self.torque_buf[:, -1] = self.get_joint_torques()

            # plot trajecs when the buffers are filled
            try: self.trajecs_recorded += 1
            except: self.trajecs_recorded = 1
            if self.trajecs_recorded % (_trajec_buffer_length) == 0:
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
        names = ['Simulation [rad]'] # line names (legend)
        second_y_axis_pos = 1.0

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
            plt.title(self.kinem_labels[i_joint])
        lines.append(line[0])

        # plot ref trajecs
        PLOT_REFS = True
        if PLOT_REFS:
            trajecs = self.trajecs_buffer[1,:,:]
            # copy ankle joint trajectories todo: remove
            # trajecs[5, :] = trajecs[8, :]
            for i_joint in range(num_joints):
                line = axes[i_joint].plot(trajecs[i_joint, :])
            lines.append(line[0])
            names.append('Reference [rad]')

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

        PLOT_TORQUES = True
        if PLOT_TORQUES:
            plot_actions(self.torque_buf/1000, "Joint Torque [kNm]")
            second_y_axis_pos = 1.12

        PLOT_ACTIONS = True
        if PLOT_ACTIONS:
            plot_actions(self.action_buf, 'PD Target [rad]', '#ff0000')

        # plot the legend in a separate subplot
        with sns.axes_style("white", {"axes.edgecolor": 'white'}):
            legend_subplot = plt.subplot(rows, cols, num_joints + 2)
            legend_subplot.set_xticks([])
            legend_subplot.set_yticks([])
            legend_subplot.legend(lines, names, loc='lower right')

        # fix title overlapping when tight_layout is true
        plt.gcf().tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.55, hspace=0.5)
        plt.suptitle('Simulation and Reference Joint Kinematics over Time '
                     '(Angles in [rad], Angular Velocities in [rad/s])')


        # add rewards
        plt.subplot(rows, cols, len(axes)+1, sharex=axes[-1])
        plt.plot(self.rewards[-_trajec_buffer_length:])
        plt.vlines(np.argwhere(self.dones_buf).flatten()+1,
                   0 , 1, colors='#cccccc')
        plt.ylim([-0.775, 1.025])
        plt.title('Rewards')


        plt.show()
        raise SystemExit('Planned exit after closing trajectory comparison plot.')