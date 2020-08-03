# suppress the annoying FutureWarnings at startup
import warnings, os
warnings.filterwarnings('ignore', category=FutureWarning)

from os.path import dirname
from tensorflow import keras
from tensorflow.keras import layers
from scripts.common.utils import import_pyplot
from sklearn.model_selection import train_test_split
from scripts.behavior_cloning.dataset import get_obs_and_delta_actions

from scripts.common import config as cfg
plt = import_pyplot()

def s(digit):
    return str(digit * 10).replace('.', '')[:4]

SHUFFLE = True

LATENT_SPACE_DIM = 8
HID_DIM = 1024
EPOCHS = 250
LEARN_RATE0 = 0.001
LEARN_RATE1 = 0.00005
notes = '' #'reproducing best results so far'

hypers_string = f'HYPERS: DIM {LATENT_SPACE_DIM}, HID_DIM {HID_DIM}, LR0 {LEARN_RATE0},' \
                f'LR1 {LEARN_RATE1}, eps {EPOCHS} | {notes}'
print(hypers_string)

encoder_layer_sizes = [HID_DIM, LATENT_SPACE_DIM]
decoder_layer_sizes = [HID_DIM] # cfg.hid_layer_sizes

# insights
"""
Best results so far: 
- [512, 8, 512] LR 0.01, 40eps - 0.1
- HYPERS: DIM 8, HID_DIM 1024, LR0 0.001,LR1 0.0001, eps 200 

Deeper doesn't seem to be better. 256, 256, 8, 256, 256 was more worse than a single 256 layer.
- when using the same LR of 0.01 that was perfect for the single layer autoenc
"""

def build_autoencoder(state_dim):
    model = keras.Sequential()
    # model.add(keras.Input(shape=(state_dim,), name='input'))
    l2_coeff = 0.0005
    # build the encoder
    input_dims = [19] + encoder_layer_sizes
    for i, hid_size in enumerate(encoder_layer_sizes):
        model.add(layers.Dense(hid_size, activation='relu', name=f'enc_hid{i+1}',
                               input_dim=input_dims[i],
                               kernel_initializer=keras.initializers.Orthogonal(gain=0.01)))
                               # kernel_regularizer=keras.regularizers.l2(l=l2_coeff)))
    # build the decoder
    input_dims = [LATENT_SPACE_DIM] + encoder_layer_sizes
    for i, hid_size in enumerate(decoder_layer_sizes):
        model.add(layers.Dense(hid_size, activation='relu', name=f'dec_hid{i+1}',
                               input_dim=input_dims[i],
                               kernel_initializer=keras.initializers.Orthogonal(gain=0.01)))
                               # kernel_regularizer=keras.regularizers.l2(l=l2_coeff)))
    # build the output layer
    model.add(layers.Dense(state_dim, activation='linear', name='output',
                           kernel_initializer=keras.initializers.Orthogonal(gain=0.01)))
                           # kernel_regularizer=keras.regularizers.l2(l=l2_coeff)))
    model.summary()
    return model

if __name__ == '__main__':
    x_data, y_data = get_obs_and_delta_actions()
    # The autoencoder should reconstruct the input
    y_data = x_data

    # train, val, test split
    x_train, x_test, y_train, y_test = \
        train_test_split(x_data, y_data, test_size=0.2, shuffle=SHUFFLE)
    x_val, x_test, y_val, y_test = \
        train_test_split(x_test, y_test, test_size=0.5, shuffle=SHUFFLE)

    # check sizes
    print('X train, val, test sizes: ', x_train.shape, x_val.shape, x_test.shape)
    print('Y train, val, test sizes: ', y_train.shape, y_val.shape, y_test.shape)

    # check shuffling
    if SHUFFLE and False:
        plt.plot(x_train[:500, [3,6,9]])
        plt.show()

    # build model
    model = build_autoencoder(x_train.shape[1])

    # lr_schedule = keras.optimizers.schedules.InverseTimeDecay(LEARN_RATE)
    # compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARN_RATE0),
        # Loss function to minimize
        loss='mean_absolute_error',
        # List of metrics to monitor
        metrics=[keras.metrics.MeanSquaredError(),
                 keras.metrics.MeanSquaredLogarithmicError(),
                 keras.metrics.MeanAbsoluteError(),
                 keras.metrics.MeanAbsolutePercentageError()],
    )

    # construct save paths
    model_path = dirname(dirname(dirname(__file__))) \
                 + '/scripts/dim_reduction/models/'
    model_name = f'{LATENT_SPACE_DIM}D_ramp' + f'_hd{HID_DIM}_ep{EPOCHS}_' \
                 f'lr0{s(LEARN_RATE0)}_lr1{s(LEARN_RATE1)}'

    # create callback to save best model
    best_model_path = model_path + f'best/{model_name}'
    save_best_callback = keras.callbacks.ModelCheckpoint(
        best_model_path, save_best_only=True,
        monitor='val_loss', mode='min')


    def linear_lr_schedule(epoch):
        return LEARN_RATE0 + (LEARN_RATE1-LEARN_RATE0)/EPOCHS * epoch

    lr_decay_callback = keras.callbacks.LearningRateScheduler(linear_lr_schedule)

    # train model
    history = model.fit(x_train, y_train, epochs=EPOCHS, verbose=1,
                        validation_data=(x_val, y_val),
                        callbacks=[save_best_callback, lr_decay_callback])

    # print the model summary
    model.summary()

    # evaluate the model
    print('\nFINAL MODEL EVALUATION!')
    train_metrics = model.evaluate(x_train, y_train, verbose=0)
    test_metrics = model.evaluate(x_test, y_test, verbose=0)
    print('MAE:\nTrain: %.5f, Test: %.5f\n' % (train_metrics[0], test_metrics[0]))
    # plot loss during training
    losses = ['loss',  'mean_squared_error', 'mean_squared_logarithmic_error',
              'mean_absolute_error', 'mean_absolute_percentage_error']

    plt.rcParams.update({'figure.autolayout': False})

    for i, loss_str in enumerate(losses):
        plt.subplot(2, len(losses)//2+1, i+1)
        plt.title(str.capitalize(loss_str))
        plt.plot(history.history[loss_str], label='train')
        plt.plot(history.history['val_'+loss_str], label='validation')

    # fix title overlapping when tight_layout is true
    plt.gcf().tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.55, hspace=0.5)
    plt.suptitle(hypers_string + \
                 f' hids {encoder_layer_sizes + decoder_layer_sizes} | '
                 f'{notes}')
    plt.legend()
    plt.show()

    # save weights of best model
    best_model = keras.models.load_model(best_model_path)
    best_model_weights_path = best_model_path.replace('best', 'weights')
    best_model.save_weights(best_model_weights_path + '.h5')
    print('Saved weights of best model in:\n', best_model_weights_path)

    # evaluate also the best model
    print('\nBEST MODEL EVALUATION!')
    train_metrics = model.evaluate(x_train, y_train, verbose=0)
    test_metrics = model.evaluate(x_test, y_test, verbose=0)
    mae_train = train_metrics[0]
    mae_test = test_metrics[0]
    print('MAE:\nTrain: %.5f, Test: %.5f\n' % (mae_train, mae_test))
    # plot loss during training
    losses = ['loss', 'mean_squared_error', 'mean_squared_logarithmic_error',
              'mean_absolute_error', 'mean_absolute_percentage_error']

    # rename saved files to add MAE
    eval_str = f'_tst{s(mae_test)}_trn{s(mae_train)}'
    new_model_path = best_model_path + eval_str
    os.rename(best_model_path, new_model_path)
    new_weight_path = best_model_weights_path + eval_str + '.h5'
    os.rename(best_model_weights_path+'.h5', new_weight_path)

    print(hypers_string)
