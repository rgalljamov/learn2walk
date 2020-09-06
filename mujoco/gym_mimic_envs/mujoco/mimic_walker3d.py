import numpy as np
from gym import utils
from os.path import join, dirname
from gym.envs.mujoco import mujoco_env
from gym_mimic_envs.mimic_env import MimicEnv
from scripts.mocap import ref_trajecs as refs

# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_viewer_at_first_step = True

# qpos and qvel indices for quick access to the reference trajectories
qpos_indices = [refs.COM_POSX, refs.COM_POSY, refs.COM_POSZ,
                refs.TRUNK_ROT_Q1, refs.TRUNK_ROT_Q2, refs.TRUNK_ROT_Q3, refs.TRUNK_ROT_Q4,
                refs.HIP_FRONT_ANG_R, refs.HIP_SAG_ANG_R,
                refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                refs.HIP_FRONT_ANG_L, refs.HIP_SAG_ANG_L,
                refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

qvel_indices = [refs.COM_VELX, refs.COM_VELY, refs.COM_VELZ,
                refs.TRUNK_ANGVEL_X, refs.TRUNK_ANGVEL_Y, refs.TRUNK_ANGVEL_Z,
                refs.HIP_FRONT_ANGVEL_R, refs.HIP_SAG_ANGVEL_R,
                refs.KNEE_ANGVEL_R, refs.ANKLE_ANGVEL_R,
                refs.HIP_FRONT_ANGVEL_L, refs.HIP_SAG_ANGVEL_L,
                refs.KNEE_ANGVEL_L, refs.ANKLE_ANGVEL_L]


class MimicWalker3dEnv(mujoco_env.MujocoEnv, utils.EzPickle, MimicEnv):
    '''
    The 2D Mujoco Walker from OpenAI Gym extended to match
    the 3D bipedal walker model from Guoping Zhao.
    '''

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self,
                                      join(dirname(__file__), "assets","walker3pd.xml"), 4)
        utils.EzPickle.__init__(self)
        # init the mimic environment, automatically loads and inits ref trajectories
        global qpos_indices, qvel_indices
        MimicEnv.__init__(self, refs.ReferenceTrajectories(qpos_indices, qvel_indices))


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
        # todo: remove after tests with guopings model
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20