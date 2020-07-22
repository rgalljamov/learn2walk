from tensorflow import keras
# todo: fix: import utils before config to avoid import errors
from scripts.common import utils
from scripts.common import config as cfg
from matplotlib import pyplot as plt
from scripts.behavior_cloning.pretrain import get_normed_obs_and_delta_actions

def load_weights():
    model = load_model()
    w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out = model.get_weights()
    return w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out

def load_model():
    model_path = cfg.abs_project_path \
                 + 'scripts/behavior_cloning/models/best/deltas_norm_obs_MAE_ep200'
    model = keras.models.load_model(model_path)
    return model

def test_model():
    """
    Loads a model and compares it's predictions
    with the optimal actions derived from the reference trajectories
    """
    # get data used for training
    # x_data contains normalized observations, y_data contains deltas
    x_data, y_data = get_normed_obs_and_delta_actions()
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


if __name__ == '__main__':
    test_model()




