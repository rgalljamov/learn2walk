import numpy as np
from matplotlib import pyplot as plt

from scripts.behavior_cloning.obs_rms import get_obs_rms
from scripts.mocap import ref_trajecs as rt

# qpos and qvel indices for quick access to the reference trajectories
qpos_indices = [rt.COM_POSX, rt.COM_POSZ, rt.TRUNK_ROT_Y,
                rt.HIP_SAG_ANG_R, rt.KNEE_ANG_R, rt.ANKLE_ANG_R,
                rt.HIP_SAG_ANG_L, rt.KNEE_ANG_L, rt.ANKLE_ANG_L]

qvel_indices = [rt.COM_VELX, rt.COM_VELZ, rt.TRUNK_ANGVEL_Y,
                rt.HIP_SAG_ANGVEL_R,
                rt.KNEE_ANGVEL_R, rt.ANKLE_ANGVEL_R,
                rt.HIP_SAG_ANGVEL_L,
                rt.KNEE_ANGVEL_L, rt.ANKLE_ANGVEL_L]

def build_data_matrix(refs):
    """:returns: 2D matrix containing all the data points in the reference trajectory:
                 (LENGTH_ALL_STEPS, STATE_DIM)
    """
    # N_STEPS x (STATE_DIM, LENGTH_STEP)
    step_list = refs.data.tolist()
    n_steps = len(refs.data)
    print("Number of steps: ", n_steps)

    step_lens = [len(step[0,:]) for step in refs.data]
    n_data_points = np.sum(step_lens)
    print('Number of data points: ', n_data_points)

    data_matrix = np.concatenate(step_list, axis=1).transpose()
    print('Data Matrix Shape: ', data_matrix.shape)

    TEST = False
    if TEST:
        plt.subplot(2,1,1)
        plt.plot(data_matrix[:, rt.KNEE_ANG_L])
        plt.subplot(2,1,2)
        plt.plot(data_matrix[:, rt.KNEE_ANGVEL_L])
        plt.show()

    return np.array(data_matrix, dtype=np.float)

def get_com_vel_all_steps():
    refs = get_refs()
    data = refs.data
    com_vels = []
    mean_vels = []
    for step in range(len(data)):
        vels = data[step][15]
        mean_vel = np.mean(vels)
        print(f'{step} \t {mean_vel}')
        mean_vels.extend(np.ones_like(vels)*mean_vel)
        com_vels.extend(vels)
    return mean_vels


def get_refs(refs=None):
    if refs is None:
        refs = rt.ReferenceTrajectories(qpos_indices, qvel_indices, {})
    return refs


def get_data(refs:rt.ReferenceTrajectories=None, debug=False):
    """
    :returns:   data_x (n_points, state_dim)
                data_y (n_points, act_dim)
    """
    if refs is None:
        refs = rt.ReferenceTrajectories(qpos_indices, qvel_indices, {})
        debug = False
    x_data, y_data = [], []
    # iterate twice through the refs to get all points
    # as we're skipping 1 step in refs.next()
    for i in range(2):
        refs.reset()
        assert refs._i_step == 0 and refs._pos == 0
        # first iteration, start at pos 0, second at pos 1
        refs.next(increment=i)
        while not refs.has_reached_last_step:
            ### prepare x data
            qpos, qvel = refs.get_ref_kinmeatics()
            # remove COM x position as the action should be independent of it
            qpos = qpos[1:]
            if all(np.copy(qpos)<0.001):
                print('similar')
                stop = True
            desired_walking_speed = refs.get_step_velocity()
            phase = refs.get_phase_variable()
            obs = np.concatenate([np.array([phase, desired_walking_speed]), qpos, qvel]).ravel()
            x_data.append(obs)

            ### prepare y data
            # get the desired joint positions at the next timestep
            refs.next()
            des_qpos = refs.get_qpos()
            # extract only actuated joints
            des_qpos = np.copy(des_qpos[3:])
            assert len(des_qpos) == 6, f'6 actions expected, got {len(des_qpos)}'
            y_data.append(des_qpos)

        if debug:
            print('x_data shape: ', np.array(x_data).shape)
            print('y_data shape: ', np.array(y_data).shape)

            plt.subplot(2, 2, i+1)
            plt.plot(np.array(x_data)[:, 5])
            plt.subplot(2, 2, i+3)
            plt.plot(np.array(x_data)[:, 14])

            if i== 0:
                x_data1, y_data1 = np.array(x_data), np.array(y_data)
                x_data, y_data = [], []

    if debug:
        x_data2, y_data2 = np.array(x_data), np.array(y_data)
        plt.plot(x_data1[:, 5])
        plt.plot(x_data2[:, 5])
        plt.show()

        plt.figure()
        plt.subplot(2,2,1)
        plt.title('knee angle 2')
        plt.plot(x_data2[:,5])
        plt.subplot(2,2,3)
        plt.title('knee angle difference')
        plt.plot((x_data1[:3000]-x_data2[:3000])[:, 5])
        plt.subplot(2,2,2)
        plt.plot(y_data1[:,1])
        plt.subplot(2,2,4)
        plt.plot((y_data1[:3000]-y_data2[:3000])[:, 1])

    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data

def get_refs_stats(refs:rt.ReferenceTrajectories=None, all_joints=True, debug=False):
    """:return: mean, variance and standard deviation of all or specified joints."""
    global qpos_indices
    if refs is None:
        refs = rt.ReferenceTrajectories(qpos_indices, qvel_indices, {})
        debug = False
    data = build_data_matrix(refs)
    means = np.mean(data, axis=0)
    vars = np.var(data, axis=0)
    stds = np.std(data, axis=0)
    if not all_joints:
        # remove com x position
        qpos_indices = qpos_indices[1:]
        indices = np.concatenate([qpos_indices, qvel_indices])
        # add phase variable
        data = np.copy(data[:, indices])
        means = np.copy(means[indices])
        vars = np.copy(vars[indices])
        stds = np.copy(stds[indices])
    if debug:
        for i in range(5,13):
            plt.subplot(2,4,i-4)
            plt.plot(data[:500,i])
            mean = np.ones((500,)) * means[i]
            std = stds[i]
            x = range(500)
            plt.plot(x, mean)
            plt.fill_between(x, mean + std, mean - std,
                             facecolor='orange', alpha=0.25)

        plt.show()
    return means, vars, stds

def test_data(x_data, y_data, is_y_delta=False):
    print('x_data shape: ', np.array(x_data).shape)
    print('y_data shape: ', np.array(y_data).shape)

    x_actuated_joint_indices = np.array(range(13,19)) # get_actuated_joint_indices()
    n_points = 500
    for i_y, i_x in enumerate(x_actuated_joint_indices):
        try:
            sub = plt.subplot(2, 6, i_y + 1, sharex=sub)
        except:
            sub = plt.subplot(2, 6, i_y + 1)
        sub.plot(x_data[:n_points, i_x])
        sub.plot(y_data[:n_points, i_y])
        sub.legend(['x', 'y'])

        sub_btm = plt.subplot(2, 6, i_y + 7, sharex=sub)
        sub_btm.plot(x_data[1:n_points + 1, i_x])
        sub_btm.plot(y_data[:n_points, i_y])
        sub_btm.legend(['x shifted (+1)', 'y'])
    plt.show()

def test_deltas(x_data, y_delta):
    act_joint_vel_indices = np.array(range(13,19))
    raise NotImplementedError()

def get_actuated_joint_indices():
    x_actuated_joint_indices = np.array(range(4, 10))
    print('Actuated joints indices: ', x_actuated_joint_indices)
    return x_actuated_joint_indices

def get_delta_angs(x_data, y_data):
    act_joint_indices = get_actuated_joint_indices()
    y_delta = y_data - x_data[:, act_joint_indices]
    return y_delta

if __name__ == '__main__':
    DEBUG = False

    refs = rt.ReferenceTrajectories(qpos_indices, qvel_indices, {})
    refs._step = refs.data[0]
    # 38 dims, 262 timesteps per step approximately
    dofs, timesteps = refs._step.shape

    means, vars, stds = get_refs_stats(refs, False, True)
    print('Means and STDs shape:', (means.shape, stds.shape))
    x_data, y_data = get_data(refs, DEBUG)
    y_delta = get_delta_angs(x_data, y_data)
    test_data(x_data, y_data)


def get_normed_obs_and_delta_actions():
    """
    Loads the SL data extracted from the reference trajectories,
    calculates the action deltas and normalize the observations.
    """
    # get data
    x_data, y_data = get_data()
    # test_data(x_data, y_data)
    y_data = get_delta_angs(x_data, y_data)
    x_mean, x_var = get_obs_rms(True)
    # normalize x_data by mean and var
    x_data_normed = (x_data - x_mean) / np.sqrt(x_var + 1e-4)

    # print('shape x_data: ', x_data.shape)
    # print('shape x_data normed: ', x_data_normed.shape)
    # print('x_data: ',x_data[:5,10:13])
    # print('x_data normed: ',x_data_normed[:5,10:13])
    # plot the normalized and unnormalized data
    # plt.plot(x_data[:500, 5])
    # plt.plot(x_data_normed[:500, 5])
    # plt.show()
    # exit(33)

    assert (np.abs((x_data- x_data_normed))>0.0000001).all(), \
        'Observation Normalization had no effect!'
    return x_data_normed, y_data