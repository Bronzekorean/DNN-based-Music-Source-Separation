import keras
from consts import CONSTS
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def make_amplitude_layer():
    amplitude_input = keras.layers.Input(shape=(CONSTS.FFT_BINS, 2, 11))
    amplitude_layer = keras.layers.Flatten()(amplitude_input)
    amplitude_layer = keras.layers.Dense(500, activation='relu')(amplitude_layer)
    amplitude_layer = keras.layers.Dense(500, activation='relu')(amplitude_layer)
    return amplitude_input, amplitude_layer


def make_phase_layer():
    phase_input = keras.layers.Input(shape=(CONSTS.FFT_BINS, 4, 11,))
    phase_layer = keras.layers.Flatten()(phase_input)
    phase_layer = keras.layers.Dense(500, activation='relu')(phase_layer)
    phase_layer = keras.layers.Dense(500, activation='relu')(phase_layer)
    return phase_input, phase_layer


def make_compile_model(name, loss=root_mean_squared_error):
    amplitude_input, amplitude_layer = make_amplitude_layer()
    phase_input, phase_layer = make_phase_layer()
    output = keras.layers.concatenate([amplitude_layer, phase_layer])
    output = keras.layers.Dense(CONSTS.FFT_BINS * 2, activation='relu')(output)
    output = keras.layers.Reshape((CONSTS.FFT_BINS, 2))(output)

    model = keras.models.Model(inputs=[amplitude_input, phase_input], outputs=output, name=name)

    model.compile(loss=loss, optimizer='adam')

    return model

