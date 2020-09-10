import numpy as np
from gym import utils
from os.path import join, dirname
from gym.envs.mujoco import mujoco_env
from gym_mimic_envs.mimic_env import MimicEnv

from scripts.common import config as cfg
from scripts.mocap import ref_trajecs as refs

# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_viewer_at_first_step = True

# qpos and qvel indices for quick access to the reference trajectories
qpos_indices = [refs.COM_POSX, refs.COM_POSY, refs.COM_POSZ,
                refs.TRUNK_ROT_X, refs.TRUNK_ROT_Y, refs.TRUNK_ROT_Z,
                refs.HIP_SAG_ANG_R, refs.HIP_FRONT_ANG_R,
                refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                refs.HIP_SAG_ANG_L, refs.HIP_FRONT_ANG_L,
                refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

qvel_indices = [refs.COM_VELX, refs.COM_VELY, refs.COM_VELZ,
                refs.TRUNK_ANGVEL_X, refs.TRUNK_ANGVEL_Y, refs.TRUNK_ANGVEL_Z,
                refs.HIP_SAG_ANGVEL_R, refs.HIP_FRONT_ANGVEL_R,
                refs.KNEE_ANGVEL_R, refs.ANKLE_ANGVEL_R,
                refs.HIP_SAG_ANGVEL_L, refs.HIP_FRONT_ANGVEL_L,
                refs.KNEE_ANGVEL_L, refs.ANKLE_ANGVEL_L]


class MimicWalker3dEnv(MimicEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    '''
    The 2D Mujoco Walker from OpenAI Gym extended to match
    the 3D bipedal walker model from Guoping Zhao.
    '''

    def __init__(self):
        walker_xml = {'mim3d': 'walker3pd.xml',
                      'mim_trq3d': 'walker3d.xml'}[cfg.env_name]
        mujoco_env.MujocoEnv.__init__(self,
                                      join(dirname(__file__), "assets", walker_xml), 4)
        utils.EzPickle.__init__(self)
        # init the mimic environment, automatically loads and inits ref trajectories
        global qpos_indices, qvel_indices
        MimicEnv.__init__(self, refs.ReferenceTrajectories(qpos_indices, qvel_indices))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

        # ----------------------------
        # Methods we override:
        # ----------------------------

    def _get_COM_indices(self):
        """
        Needed to distinguish between joint and COM kinematics.

        Returns a list of indices pointing at COM joint position/index
        in the considered robot model, e.g. [0,1,2]
        """
        return [0,1,2]

    def _get_saggital_trunk_joint_index(self):
        return 4

    def _get_not_actuated_joint_indices(self):
        """
        Needed for playing back reference trajectories
        by using position servos in the actuated joints.

        @returns a list of indices specifying indices of
        joints in the considered robot model that are not actuated.
        Example: return [0,1,2]
        """
        return self._get_COM_indices() + [3,4,5]

    def _get_max_actuator_velocities(self):
        """Maximum joint velocities approximated from the reference data."""
        return np.array([5, 1, 10, 10, 5, 1, 10, 10])

    def has_ground_contact(self):
        has_contact = [False, False]
        for contact in self.data.contact[:10]:
            if contact.geom1 == 0 and contact.geom2 == 4:
                # right foot has ground contact
                has_contact[1] = True
            elif contact.geom1 == 0 and contact.geom2 == 7:
                # left foot has ground contact
                has_contact[0] = True
        if cfg.is_mod(cfg.MOD_3_PHASES):
            has_contact += [all(has_contact)]
        return has_contact