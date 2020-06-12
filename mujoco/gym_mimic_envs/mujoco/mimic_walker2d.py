import numpy as np
from gym import utils
from os.path import join, dirname
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

qvel_indices = [refs.COM_VELX, refs.COM_VELZ, refs.TRUNK_ANGVEL_Y,
                refs.HIP_SAG_ANGVEL_R,
                refs.KNEE_ANGVEL_R, refs.ANKLE_ANGVEL_R,
                refs.HIP_SAG_ANGVEL_L,
                refs.KNEE_ANGVEL_L, refs.ANKLE_ANGVEL_L]

# adaptations needed to account for different body shape
# and axes definitions in the reference trajectories
ref_trajec_adapts = {}


class MimicWalker2dEnv(MimicEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Building upon the Walker2d-v2 Environment with the id: Walker2d-v2
    """

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self,
                                      join(dirname(__file__), "assets", 'walker2d.xml'), 4)
        utils.EzPickle.__init__(self)
        # init the mimic environment, automatically loads and inits ref trajectories
        MimicEnv.__init__(self, RefTrajecs(qpos_indices, qvel_indices, ref_trajec_adapts))
        self.model.opt.timestep = 1e-3
        self.frame_skip = 5

    def step(self, a):
        # pause sim on startup to be able to change rendering speed, camera perspective etc.
        global pause_viewer_at_first_step
        if pause_viewer_at_first_step:
            self._get_viewer('human')._paused = True
            pause_viewer_at_first_step = False

        MimicEnv.step(self)

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

        USE_DMM_REW = True
        if USE_DMM_REW:
            reward = self.get_imitation_reward()
            # print('Reward ', reward)
        else:
            reward = (com_x_vel_finite_difs)
            reward += alive_bonus
            reward -= 1e-3 * np.square(a).sum()

        USE_ET = False
        USE_REW_ET = True
        if USE_ET:
            done = self.has_exceeded_allowed_deviations()
        elif USE_REW_ET:
            done = self.do_terminate_early(reward, height, ang)
        else:
            done = not (height > 0.8 and height < 2.0 and
                        ang > -1.0 and ang < 1.0)
        # if done: print('Done')
        ob = self._get_obs()
        return ob, reward, done, {}

    def reset_model(self):
        return MimicEnv.reset_model(self)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    # ----------------------------
    # Methods we override:
    # ----------------------------

    def _get_COM_indices(self):
        return [0,1]

    def _get_not_actuated_joint_indices(self):
        return [0,1,2]