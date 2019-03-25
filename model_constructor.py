import keras
from consts import CONSTS
from keras import backend as K

def make_amplitude_layer(mean=CONSTS.AMPLITUDE_MEAN , std=CONSTS.AMPLITUDE_STD):
    """ creates keras first layer that takes amplitude as input
        Returns:
        amplitude_input: keras input layer takes amplitude
        amplitude_layer: karas architecture untill concatenation"""
    amplitude_input = keras.layers.Input(shape=(CONSTS.FFT_BINS, 2, 11))
    amplitude_layer = keras.layers.Flatten()(amplitude_input)
    amplitude_layer = keras.layers.Lambda(lambda x: (x - mean)/std)(amplitude_layer)
    amplitude_layer = keras.layers.Dense(500, activation='relu')(amplitude_layer)
    amplitude_layer = keras.layers.Dense(500, activation='relu')(amplitude_layer)
    return amplitude_input, amplitude_layer


def make_phase_layer():
    """ creates keras first layer that takes phase derivatives as input
        Returns:
        phase_input: keras input layer takes phase derivatives
        phase_layer: karas architecture untill concatenation"""
    phase_input = keras.layers.Input(shape=(CONSTS.FFT_BINS, 4, 11,))
    phase_layer = keras.layers.Flatten()(phase_input)
    phase_layer = keras.layers.Dense(500, activation='relu')(phase_layer)
    phase_layer = keras.layers.Dense(500, activation='relu')(phase_layer)
    return phase_input, phase_layer


def make_compile_model(name, loss='mean_squared_error'):
    """ concatenate phase and amplitude layer, creates the full architecture and compiles using loss
        Returns:
        model: compiles model"""
    amplitude_input, amplitude_layer = make_amplitude_layer()
    phase_input, phase_layer = make_phase_layer()
    output = keras.layers.concatenate([amplitude_layer, phase_layer])
    output = keras.layers.Dense(CONSTS.FFT_BINS * 2, activation='relu')(output)
    output = keras.layers.Reshape((CONSTS.FFT_BINS, 2))(output)

    model = keras.models.Model(inputs=[amplitude_input, phase_input], outputs=output, name=name)

    model.compile(loss=loss, optimizer='adam')

    return model

