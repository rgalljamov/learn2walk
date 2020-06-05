'''
Script to handle reference trajectories.
- Get trajectory for a single step
- Identify most appropriate next step
- Get only specified trajectory parts

- handle COM and Joint Kinematics separately
'''

import numpy as np
import scipy.io as spio

# reference trajectory: joint position indices
COM_POSX, COM_POSY, COM_POSZ = range(0,3)
TRUNK_ROT_Q1, TRUNK_ROT_Q2, TRUNK_ROT_Q3, TRUNK_ROT_Q4 = range(3,7)
HIP_FRONT_ANG_R, HIP_SAG_ANG_R, KNEE_ANG_R, ANKLE_ANG_R = range(7,11)
HIP_FRONT_ANG_L, HIP_SAG_ANG_L, KNEE_ANG_L, ANKLE_ANG_L = range(11,15)

# reference trajectory: joint velocity indices
COM_VELX, COM_VELY, COM_VELZ = range(15,18)
TRUNK_ANGVEL_X, TRUNK_ANGVEL_Y, TRUNK_ANGVEL_Z = range(18,21)
HIP_FRONT_ANGVEL_R, HIP_SAG_ANGVEL_R, KNEE_ANGVEL_R, ANKLE_ANGVEL_R = range(21,25)
HIP_FRONT_ANGVEL_L, HIP_SAG_ANGVEL_L, KNEE_ANGVEL_L, ANKLE_ANGVEL_L = range(25,29)

# reference trajectory: foot position and GRF indices
FOOT_POSX_L, FOOT_POSY_L, FOOT_POSZ_L, FOOT_POSX_R, FOOT_POSY_R, FOOT_POSZ_R = range(29,35)
GRF_R, GRF_L = range(35,37)
TRUNK_ROT_X, TRUNK_ROT_Y, TRUNK_ROT_Z = range(37,40)

# on my local PC
PATH_REF_TRAJECS = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/' \
                   'assets/ref_trajecs/Trajecs_Ramp_Slow_200Hz_EulerTrunkAdded.mat'

PATH_TRAJEC_RANGES = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/' \
                     'assets/ref_trajecs/Trajec_Ranges_Ramp_Slow_200Hz_EulerTrunkAdded.npz'

# todo: automatically detect LL PC
REMOTE = False

# executing script on the Lauflabor PC
if REMOTE:
    PATH_REF_TRAJECS = '/home/rustam/code/remote/' \
                       'assets/ref_trajecs/Trajecs_Ramp_Slow_200Hz_EulerTrunkAdded.mat'

    PATH_TRAJEC_RANGES = '/home/rustam/code/remote/' \
                       'assets/ref_trajecs/Trajec_Ranges_Ramp_Slow_200Hz_EulerTrunkAdded.npz'


class ReferenceTrajectories:

    def __init__(self, qpos_indices, q_vel_indices, adaptations={}):
        self.path = PATH_REF_TRAJECS
        self.qpos_is = qpos_indices
        self.qvel_is = q_vel_indices
        # data contains 250 steps consisting of 40 trajectories
        self.data = self._load_trajecs()
        self._adapt_trajecs_to_other_body(adaptations)
        # calculated and added trunk euler rotations
        # self._add_trunk_euler_rotations()
        # calculate ranges needed for Early Termination
        self.ranges = self._determine_trajectory_ranges()
        # current step
        self.step = self._get_random_step()
        # passed time before the current step was chosen
        self._step_start_time = 0
        # position on the current reference step trajectory
        self.pos = 0
        # distance walked so far (COM X Position)
        self.dist = 0

    def get_qpos(self, timestep):
        return self.get_by_indices(self.qpos_is, timestep)

    def get_qvel(self, timestep):
        return self.get_by_indices(self.qvel_is, timestep)


    def get_kinematic_ranges(self):
        '''Returns the maximum range of qpos and qvel in reference trajecs.'''
        return self.ranges[self.qpos_is], self.ranges[self.qvel_is]
    def _adapt_trajecs_to_other_body(self, adapts: dict):
        '''The trajectories were collected from a single reference person.
           They have to be adjusted when used with a model
           with different body properties compared to the reference person.'''
        indices = adapts.keys()
        for index in indices:
            for i_step in range(len(self.data)):
                self.data[i_step][index,:] *= adapts[index]

    def get_by_indices(self, indices, timestep):
        # after a first step was taken, we have to set the timestep to 0 again
        self.pos = timestep - self._step_start_time
        if self.pos == len(self.step[0]):
            self._step_start_time = timestep
            self.step = self.get_next_step()
            self.pos = 0
        return self.step[indices, self.pos]


    def get_random_init_state(self):
        ''' Random State Initialization:
            @returns: qpos and qvel of a random step at a random position'''
        self.step = self._get_random_step()
        self.pos = np.random.randint(0, len(self.step[0]) - 1)
        return self.get_qpos(self.pos), self.get_qvel(self.pos)


    def get_com_kinematics(self):
        com_pos = self.step[:3,:]
        com_vel = self.step[15:18,:]
        return com_pos, com_vel

    def get_trunk_rotation(self):
        ''':returns trunk_rot: in quaternions (4D)
                    trun_ang_vel: corresponding angular velocities (3D)'''
        trunk_rot = self.step[3:7,:]
        trunk_ang_vel = self.step[18:21,:]
        return trunk_rot, trunk_ang_vel

    def get_hip_kinematics(self):
        indices_angs = [7,8, 11,12]
        indices_vels = [21,22, 25,26]
        ang_front_r, ang_sag_r, ang_front_l, ang_sag_l = self.step[indices_angs]
        vel_front_r, vel_sag_r, vel_front_l, vel_sag_l = self.step[indices_vels]
        return ang_front_r, ang_sag_r, ang_front_l, ang_sag_l, \
               vel_front_r, vel_sag_r, vel_front_l, vel_sag_l

    def get_knee_kinematics(self):
        indices = [9, 13, 23,27]
        ang_r, ang_l, vel_r, vel_l = self.step[indices]
        return ang_r, ang_l, vel_r, vel_l

    def get_ankle_kinematics(self):
        indices = [10,14, 24,28]
        ang_r, ang_l, vel_r, vel_l = self.step[indices]
        return ang_r, ang_l, vel_r, vel_l

    def _load_trajecs(self):
        # load matlab data, containing trajectories of 250 steps
        data = spio.loadmat(self.path)
        # 250 steps, shape (250,1), where 1 is an array with kinematic data
        data = data['Data']
        # flatten the array to have dim (steps,)
        data = data.flatten()
        return data

    def _get_random_step(self):
        # which of the 250 steps are we looking at
        self.i_step = np.random.randint(0, len(self.data) - 1)
        return self.data[self.i_step]

    def get_next_step(self):
        '''The steps are sorted. To get the next step, we just have to increase the index.
           However, the COM X Position is zero'ed for each step.
           Thus, we need to add the so far traveled distance to COM X Position.'''

        # increase the step index, reset if last step was reached
        if self.i_step > len(self.data)-1:
            self.i_step = 0
        else:
            self.i_step += 1

        # update the so far traveled distance
        self.dist = self.step[COM_POSX,-1].flatten()
        # choose the next step
        step = self.data[self.i_step]
        # add the so far traveled distance to the x pos of the COM
        step[COM_POSX,:] += self.dist
        return step


    def _add_trunk_euler_rotations(self):
        '''Used to extend reference data with euler rotations of the trunk.
           Before, trunk rotations were only given in unit quaternions.'''
        from scipy.spatial.transform import Rotation as Rot

        data_dict = spio.loadmat(self.path)
        data = data_dict['Data']
        new_data = np.ndarray(data.shape, dtype=np.object)
        data = data.flatten()

        # iterate over all steps and add three more dimensions containing trunk euler rotations
        for i, step in enumerate(data):
            # get trunk rotation in quaternions: q1...q4
            q1, q2 = step[TRUNK_ROT_Q1,:], step[TRUNK_ROT_Q2,:]
            q3, q4 = step[TRUNK_ROT_Q3,:], step[TRUNK_ROT_Q4,:]
            # quaternion in scalar-last (x, y, z, w) format.
            old_rot_quat = np.array([q2, q3, q4, q1])
            rot_quat = Rot.from_quat(old_rot_quat.transpose())
            # convert to euler rotations
            euler_x, euler_y, euler_z = rot_quat.as_euler('xyz').transpose()
            # save the new angles in the reference trajectory data
            dims, dur = step.shape
            new_step = np.ndarray((dims + 3, dur), dtype=np.object)
            new_step[0:dims,:] = step
            new_step[dims:,:] = [euler_x, euler_y, euler_z]

            new_data[i,0] = new_step

        self.data = new_data.flatten()

        data_dict['Data'] = new_data
        print('BEFORE saved final')
        spio.savemat('Trajecs_Ramp_Slow_200Hz_EulerTrunkAdded.mat', data_dict, do_compression=True)
        print('saved final')
        raise SystemExit('This function was only used to transform '
                         'the Trunk Quaternion to Euler Rotations.\n'
                         'The transformed data was saved.\n'
                         'This method is now only required for documentation.')


    def _determine_trajectory_ranges(self):
        '''Needed for early termination. We terminate an episode when the agent
           deviated too much from the reference trajectories. How much deviation is allowed
           depends on the maximum range of a joint position or velocity.'''
        # load already determined and saved ranges or calculate and save if not yet happened
        try:
            npz = np.load(PATH_TRAJEC_RANGES)
            return npz['ranges']
        except FileNotFoundError:
            print('COULD NOT LOAD TRAJEC RANGES, (RE)CALCULATING THEM!')
            pass

        mins = np.ones((len(self.data),self.data[0].shape[0]))
        maxs = np.ones_like(mins)
        for i_step, step in enumerate(self.data):
            for i_traj, traj in enumerate(step):
                min = np.min(traj)[0][0]
                max = np.max(traj)[0][0]
                mins[i_step, i_traj] = min
                maxs[i_step, i_traj] = max
        mins = np.min(mins, axis=0)
        maxs = np.max(maxs, axis=0)
        ranges = maxs - mins
        np.savez(PATH_TRAJEC_RANGES, mins=mins, maxs=maxs, ranges=ranges)
        self.ranges = ranges






if __name__ == '__main__':
    refs = ReferenceTrajectories([],[])
