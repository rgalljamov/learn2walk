from tensorflow import keras
# todo: fix: import utils before config to avoid import errors
from scripts.common import utils
from scripts.common import config as cfg

def load_weights():
    model_path = cfg.abs_project_path \
                 + 'scripts/behavior_cloning/models/best/deltas_norm_obs_MAE_ep20'
    model = keras.models.load_model(model_path)
    w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out = model.get_weights()
    return w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out

if __name__ == '__main__':
    load_weights()

