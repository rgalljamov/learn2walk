"""
Optimizes the PD gains of the MimicWalker2D Env using optuna
ATTENTION:
    + change ref_trajecs.get_random_init_state() to return a fixed step at position 0
    + set PD_TUNING = True in monitor.py
    + we've chosen the best parameters based on reward weights 6400
"""
import gym, optuna, numpy as np
# necessary to import custom gym environments
import gym_mimic_envs
from gym_mimic_envs.monitor import Monitor
from scripts.common import config as cfg


RENDER = False

# set config params
cfg.modification = cfg.MOD_CUSTOM_POLICY
cfg.rew_weights = "4600"
print(f'1000, 5 FSkip')
print('Random Initialization')
print('reward weights:', cfg.rew_weights)

env = gym.make('MimicWalker2d-v0')
env = Monitor(env)

if not isinstance(env, Monitor):
    # VecNormalize wrapped DummyVecEnv
    vec_env = env
    env = env.venv.envs[0]

obs = env.reset()
env.do_fly()
env.activate_evaluation()

def objective(trial: optuna.Trial):
    k_hip = trial.suggest_uniform('k_hip', 0, 4000)
    k_knee = trial.suggest_uniform('k_knee', 0, 4000)
    k_ankle = trial.suggest_uniform('k_ankle', 0, 4000)
    d_hip = trial.suggest_uniform('d_hip', 0, 8)
    d_knee = trial.suggest_uniform('d_knee', 0, 6)
    d_ankle = trial.suggest_uniform('d_ankle', 0, 3)
    gains = [k_hip, k_knee, k_ankle] * 2
    dampings = [d_hip, d_knee, d_ankle] * 2

    # set pd gains
    env.model.actuator_gainprm[:,0] = gains
    env.model.dof_damping[3:] = dampings

    # collect rewards as evaluation metric
    rewards = []

    env.reset()
    for i in range(2000):
        # follow desired trajecs with PD Position Controllers
        des_qpos = env.get_ref_qpos(exclude_not_actuated_joints=True)
        obs, reward, done, _ = env.step(des_qpos)
        rewards.append(reward)
        if RENDER: env.render()

    return np.sum(rewards)

# optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

env.close()
