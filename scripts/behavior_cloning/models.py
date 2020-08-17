from tensorflow import keras
# todo: fix: import utils before config to avoid import errors
from scripts.common import utils
from scripts.common import config as cfg
from matplotlib import pyplot as plt
from stable_baselines import PPO2
from scripts.behavior_cloning.dataset import get_obs_and_delta_actions

def load_weights():
    import h5py
    if cfg.is_mod(cfg.MOD_FLY):
        weights_file = h5py.File(cfg.abs_project_path
                                 + 'scripts/behavior_cloning/models/best/'
                                   'MAE_const_ortho_l2_actnormFLY_ep200', 'r')
    else:
        weights_file = h5py.File(cfg.abs_project_path
                                 +'scripts/behavior_cloning/models/'
                                  'weights/MAE_ramp_ortho_l2_actnorm_ep200', 'r')
    keys = list(weights_file.keys())
    if 'model_weights' in keys:
        weights_file = weights_file['model_weights']
        keys = list(weights_file.keys())
    assert keys == ['hid1', 'hid2', 'output'], f'Layer names were: {keys}'
    hid_keys = list(weights_file['hid1'].keys())
    hid1 = weights_file['hid1']['hid1']
    # output of weights_file['hid1']['hid1_1'].keys():
    b_key, w_key = ['bias:0', 'kernel:0']
    ws = [weights_file[key][key][w_key].value for key in keys]
    bs = [weights_file[key][key][b_key].value for key in keys]
    return ws, bs

    model = load_model()
    w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out = model.get_weights()
    model = None
    return w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out

def load_model():
    model_path = cfg.abs_project_path \
                 + 'scripts/behavior_cloning/models/best/deltas_norm_obs_MAE_ep200'
    model = keras.models.load_model(model_path)
    return model


def load_encoder_weights():
    import h5py
    weights_file = h5py.File(cfg.abs_project_path
                             +'scripts/dim_reduction/models/weights/'
                              '8D_ramp_hd1024_ep200_lr0001_lr10001_tst0425_trn0417.h5', 'r')
    keys = list(weights_file.keys())
    relevant_keys = ['enc_hid1', 'enc_hid2']
    assert all([(key in keys) for key in relevant_keys])
    # output of weights_file['hid1']['hid1_1'].keys():
    b_key, w_key = ['bias:0', 'kernel:0']
    ws = [weights_file[key][key+'_1'][w_key].value for key in relevant_keys]
    bs = [weights_file[key][key+'_1'][b_key].value for key in relevant_keys]
    return ws, bs


def test_model():
    """
    Loads a model and compares it's predictions
    with the optimal actions derived from the reference trajectories
    """
    # get data used for training
    # x_data contains normalized observations, y_data contains deltas
    x_data, y_data = get_obs_and_delta_actions()
    x_data, y_data = x_data[:2000], y_data[:2000]

    plt.subplot(131)
    plt.title('Reference Data')
    refs_knee = x_data[:, 6]
    acts_knee = y_data[:, 2]
    plt.plot(refs_knee, label='knee ref angs')
    plt.plot(acts_knee * 10, label='knee acts (ang deltas) [x10]')
    plt.plot(refs_knee + acts_knee, label='knee target angs')
    plt.legend()

    # get the model prediction of delta angles
    model = load_model()
    pred_knee_deltas = model.predict(x_data)[:, 2]

    plt.subplot(132)
    plt.title('Predicted Data')
    plt.plot(refs_knee, label='knee ref angs')
    plt.plot(pred_knee_deltas * 10, label='predicted knee acts (ang deltas) [x10]')
    plt.plot(refs_knee + pred_knee_deltas, label='predicted knee target angs')
    plt.legend()

    # compare prediction with desired target actions
    plt.subplot(133)
    plt.title('refs actions vs. predictions')
    plt.plot(acts_knee * 10, label='knee acts (ang deltas) [x10]')
    plt.plot(pred_knee_deltas * 10, label='predicted knee acts (ang deltas) [x10]')
    plt.legend()
    plt.show()

def get_ppo2_weights(model_path):
    weights = []
    biases = []

    model = PPO2.load(load_path=model_path)

    for param in model.params:
        if 'pi' in param.name:
            if 'w:0' in param.name:
                weights.append(model.sess.run(param))
            elif 'b:0' in param.name:
                biases.append(model.sess.run(param))
    return weights, biases

def compare_bc_model_with_ppo_init_model():
    """Compare the pretrained model with the ppo2 model that was saved before training."""
    PATH = "/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/cstm_pi/" \
           "pretrain_pi/pi_deltas/norm_acts/mim2d/16envs/ppo2/4mio/18"
    if not PATH.endswith('/'): PATH += '/'
    checkpoint = 0

    # load model
    model_path = PATH + f'models/model_{checkpoint}.zip'
    ppo_ws, ppo_bs = get_ppo2_weights(model_path)
    ws, bs = load_weights()

    debug = True


if __name__ == '__main__':
    # test_model()
    # load_weights()
    # compare_bc_model_with_ppo_init_model()
    load_encoder_weights()


