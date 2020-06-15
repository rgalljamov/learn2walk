import gym
from gym_mimic_envs.mimic_env import MimicEnv
import numpy as np

# length of the buffer containing sim and ref trajecs for comparison
_trajec_buffer_length = 2000

class Monitor(gym.Wrapper):

    def __init__(self, env: MimicEnv):
        self.env = env
        super(Monitor, self).__init__(self.env)
        self.setup_containers()

    def setup_containers(self):
        self.ep_len = 0
        self.rewards = []
        self.returns = []
        self.ep_lens = []
        self.actions = []
        self.grfs_left = []
        self.grfs_right = []

        # monitor sim and ref trajecs for comparison (sim/ref, kinem_indices, timesteps)
        self.trajecs_buffer = np.zeros((2, len(self.num_dofs), _trajec_buffer_length))
        # monitor episode terminations
        self.dones_buf = np.zeros((_trajec_buffer_length,))

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

            # test
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
        plt = self.env.refs.plt
        plt.rcParams.update({'figure.autolayout': False})

        num_joints = len(self.kinem_labels)
        cols = 5
        rows = int((num_joints+1)/cols) + 1
        # plot sim trajecs
        trajecs = self.trajecs_buffer[0,:,:]
        axes = []
        for i_joint in range(num_joints):
            try: axes.append(plt.subplot(rows, cols, i_joint + 1, sharex=axes[i_joint-1]))
            except: axes.append(plt.subplot(rows, cols, i_joint + 1))
            trajec = trajecs[i_joint, :]
            plt.plot(trajec)
            # show episode ends
            plt.vlines(np.argwhere(self.dones_buf).flatten()+1,
                       np.min(trajec), np.max(trajec), colors='#cccccc')
            plt.title(self.kinem_labels[i_joint])
        # plot ref trajecs
        PLOT_REFS = True
        if PLOT_REFS:
            trajecs = self.trajecs_buffer[1,:,:]
            # copy ankle joint trajectories todo: remove
            # trajecs[5, :] = trajecs[8, :]
            for i_joint in range(num_joints):
                axes[i_joint].plot(trajecs[i_joint, :])
        plt.legend(['Simulation', 'Reference'], loc='lower right', bbox_to_anchor=(1.75, 0.1))
        plt.suptitle('Comparison of Simulation and Reference Joint Kinematics over Time')
        # fix title overlapping when tight_layout is true
        plt.gcf().tight_layout(rect=[0, 0, 1, 0.95])

        # add rewards
        plt.subplot(rows, cols, len(axes)+1, sharex=axes[-1])
        plt.plot(self.rewards[-_trajec_buffer_length:])
        plt.vlines(np.argwhere(self.dones_buf).flatten()+1,
                   0 , 1, colors='#cccccc')
        plt.title('Rewards')

        plt.show()