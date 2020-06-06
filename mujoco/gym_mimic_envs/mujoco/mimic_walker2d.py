import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_mimic_envs.mimic_env import MimicEnv
from scripts.common.utils import is_remote
from scripts.common import ref_trajecs as refs
from scripts.common.ref_trajecs import ReferenceTrajectories as RefTrajecs


# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_viewer_at_first_step = True and not is_remote()

# qpos and qvel indices for quick access to the reference trajectories
qpos_indices = [refs.COM_POSX, refs.COM_POSZ, refs.TRUNK_ROT_Y,
                refs.HIP_SAG_ANG_R, refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                refs.HIP_SAG_ANG_L, refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

qvel_indices = [refs.COM_VELX, refs.COM_VELY, refs.TRUNK_ANGVEL_Y,
                refs.HIP_SAG_ANGVEL_R,
                refs.KNEE_ANGVEL_R, refs.ANKLE_ANGVEL_R,
                refs.HIP_SAG_ANGVEL_L,
                refs.KNEE_ANGVEL_L, refs.ANKLE_ANGVEL_L]

# adaptations needed to account for different body shape
# and axes definitions in the reference trajectories
ref_trajec_adapts = {refs.COM_POSZ: 1.25/1.08, # difference between COM heights
                     refs.HIP_SAG_ANG_R: -1, refs.HIP_SAG_ANG_L: -1,
                     refs.HIP_SAG_ANGVEL_R: -1, refs.HIP_SAG_ANGVEL_L: -1,
                     refs.KNEE_ANG_R: -1, refs.KNEE_ANG_L: -1,
                     refs.KNEE_ANGVEL_R: -1, refs.KNEE_ANGVEL_L: -1,
                     refs.ANKLE_ANG_R: -1, refs.ANKLE_ANG_L: -1,
                     refs.ANKLE_ANGVEL_R: -1, refs.ANKLE_ANGVEL_L: -1,}


class MimicWalker2dEnv(MimicEnv, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        # init the mimic environment, automatically loads and inits ref trajectories
        global qpos_indices, qvel_indices
        MimicEnv.__init__(self, RefTrajecs(qpos_indices, qvel_indices, ref_trajec_adapts))
        self.model.opt.timestep = 1e-3
        self.frame_skip = 5

    def step(self, a):
        # pause sim on startup to be able to change rendering speed, camera perspective etc.
        global pause_viewer_at_first_step
        if pause_viewer_at_first_step:
            self._get_viewer('human')._paused = True
            pause_viewer_at_first_step = False

        qpos_before = np.copy(self.sim.data.qpos)
        qvel_before = np.copy(self.sim.data.qvel)

        posbefore = qpos_before[0]
        self.do_simulation(a, self.frame_skip)

        qpos_after = self.sim.data.qpos
        qvel_after = self.sim.data.qvel

        posafter, height, ang = qpos_after[0:3]

        qpos_delta0 = qpos_before[0] - qpos_after[0]
        qpos_delta = qpos_after - qpos_before

        alive_bonus = 1.0
        com_x_vel_finite_difs = (posafter - posbefore) / self.dt
        com_x_vel_qvel = self.sim.data.qvel[0]
        com_x_vel_delta = com_x_vel_finite_difs - com_x_vel_qvel
        qvel_findifs = (qpos_delta)/self.dt
        qvel_delta = qvel_after - qvel_findifs
        reward = (com_x_vel_finite_difs)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = self.is_early_termination()
        ob = self._get_obs()
        return ob, reward, done, {}

    def reset_model(self):
        return MimicEnv.reset_model(self)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20