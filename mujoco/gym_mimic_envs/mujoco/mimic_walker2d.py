import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_mimic_envs.mimic_env import MimicEnv
from scripts.common import ref_trajecs as refs

# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_viewer_at_first_step = True

# qpos and qvel indices for quick access to the reference trajectories
qpos_indices = [refs.COM_POSX, refs.COM_POSZ, refs.TRUNK_ROT_Y,
                refs.HIP_SAG_ANG_R, refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                refs.HIP_SAG_ANG_L, refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

qvel_indices = [refs.COM_VELX, refs.COM_VELY, refs.TRUNK_ANGVEL_Y,
                refs.HIP_SAG_ANGVEL_R,
                refs.KNEE_ANGVEL_R, refs.ANKLE_ANGVEL_R,
                refs.HIP_SAG_ANGVEL_L,
                refs.KNEE_ANGVEL_L, refs.ANKLE_ANGVEL_L]

ref_trajec_adapts = {refs.COM_POSZ: 1.25/1.08, # difference between COM heights
                     refs.HIP_SAG_ANG_R: -1, refs.HIP_SAG_ANG_L: -1,
                     refs.HIP_SAG_ANGVEL_R: -1, refs.HIP_SAG_ANGVEL_L: -1,
                     refs.KNEE_ANG_R: -1, refs.KNEE_ANG_L: -1,
                     refs.KNEE_ANGVEL_R: -1, refs.KNEE_ANGVEL_L: -1,
                     refs.ANKLE_ANG_R: -1, refs.ANKLE_ANG_L: -1,
                     refs.ANKLE_ANGVEL_R: -1, refs.ANKLE_ANGVEL_L: -1,}


class MimicWalker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle, MimicEnv):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        # init the mimic environment, automatically loads and inits ref trajectories
        global qpos_indices, qvel_indices
        MimicEnv.__init__(self, refs.ReferenceTrajectories(qpos_indices, qvel_indices, ref_trajec_adapts))


    def step(self, a):
        # pause sim on startup to be able to change rendering speed, camera perspective etc.
        global pause_viewer_at_first_step
        if pause_viewer_at_first_step:
            self._get_viewer('human')._paused = True
            pause_viewer_at_first_step = False

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        qpos, qvel = self.refs.get_random_init_state()
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20