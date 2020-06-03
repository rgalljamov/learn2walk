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


PATH_REF_TRAJECS = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/' \
                   'assets/ref_trajecs/Traj_Ramp_Slow_final.mat'


class ReferenceTrajectories:

    def __init__(self, mat_path=PATH_REF_TRAJECS):
        self.path = mat_path
        # data contains 250 steps consisting of 37 trajectories
        self.data = self._load_trajecs()
        # current step
        self.step = self._get_random_step()
        # passed time before the current step was chosen
        self._step_start_time = 0
        # position on the current reference step trajectory
        self.pos = 0

    def get_by_indices(self, indices, timestep):
        # after a first step was taken, we have to set the timestep to 0 again
        self.pos = timestep - self._step_start_time
        if self.pos == len(self.step[0]):
            self._step_start_time = timestep
            self.step = self._get_random_step()
            self.pos = 0
        return self.step[indices, self.pos]

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
        i_step = np.random.randint(0, len(self.data)-1)
        return self.data[i_step]

    def get_next_step(self):
        return self._get_random_step()

