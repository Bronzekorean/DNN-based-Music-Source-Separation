from consts import CONSTS
from scipy.signal import stft, istft, get_window
import numpy as np
import gc

def fourier_transform(audio_track):
    """ extract amplitde and phase from scipy stft:
        Returns:
        ammplitude: fft amplitude
        phase: fft phase"""
    _, _, transformed = stft(audio_track, fs=CONSTS.RATE, nperseg=CONSTS.WINDOW_SIZE, noverlap=CONSTS.OVERLAP, axis=0)

    amplitude, phase = np.absolute(transformed), np.angle(transformed)
    return amplitude, phase



def differentiate_rescale(phase):
    """ Make data from audio recoding
        args:
        audio_track numpy array stereo mixture shape (num_samples, 2)
        Returns:
            amplitude, phase and processed phase derivatives
        """

    dt_phase = np.diff(phase)
    dk_phase = np.diff(phase, axis=0)

    #pad the differences to have the same shape as the phase

    dt_phase = np.concatenate([np.zeros((phase.shape[0], phase.shape[1], 1)), dt_phase], axis=-1)
    dk_phase = np.concatenate([np.zeros((1, phase.shape[1], phase.shape[2])), dk_phase], axis=0)

    time_shift = 2 * np.pi * CONSTS.HOP_SIZE / CONSTS.WINDOW_SIZE * np.arange(phase.shape[0])
    dt_phase = (dt_phase.T + time_shift).T
    dk_phase = dk_phase + np.pi

    # recenter to (-pi, pi)
    dt_phase = np.mod(dt_phase + np.pi, 2 * np.pi) - np.pi
    dk_phase = np.mod(dk_phase + np.pi, 2 * np.pi) - np.pi

    return np.concatenate((dt_phase, dk_phase), axis=1)


def make_features(tracks):
    """Returns amplitude, phase  and phase derivatives of stft of track objects"""
    out = list()
    for track in tracks:
        amplitude, phase = fourier_transform(track.audio)
        out.append((amplitude,differentiate_rescale(phase)))
    return out


def pad_and_delete(feature_list, context=CONSTS.CONTEXT_SIZE):
    """ pads the current features list and deletes it
        Returns:
        times_features: original time frames number
        new_features_list: list of padded features
    """
    time_frames_list = list()
    new_features_list = list()
    for features in feature_list:
        time_frames = features[0].shape[-1]
        time_frames_list.append(time_frames)
        new_features = list()
        for feature in features:
            feature_shape = feature.shape
            new_features.append(np.concatenate([np.zeros(feature_shape[:2] + (context, )), feature, np.zeros(feature_shape[:2] + (context,))], axis=-1))
        new_features_list.append(new_features)
    del feature_list; gc.collect()
    return time_frames_list, new_features_list


def make_train_input(tracks, context_length=CONSTS.CONTEXT_SIZE):
    time_frames_list, padded_features = pad_and_delete(make_features(tracks), context_length)
    amplitude_out = list()
    phase_out = list()
    for time, features in zip(time_frames_list, padded_features):
        amplitude, phase_derivatives = features
        for i in range(time):
            amplitude_out.append(amplitude[:, :, i: i + 2 * context_length + 1])
            phase_out.append(phase_derivatives[:, :, i: i + 2 * context_length + 1])
    return [amplitude_out, phase_out]


def make_target_data(tracks, target):
    amplitude_out = None
    empty = True
    for track in tracks:
        amplitude, _ = fourier_transform(track.targets[target].audio)
        if empty:
            amplitude_out = amplitude
            empty = False
        else:
            amplitude_out = np.concatenate([amplitude_out, amplitude], axis=-1)

    amplitude_out = amplitude_out.swapaxes(0, 2)
    amplitude_out = amplitude_out.swapaxes(1, 2)
    return amplitude_out


def reconstruct(out_amplitude, out_phase):
    """reconstruct output from predicted amplitude and phase"""
    return istft(out_amplitude * np.exp(1j * out_phase, fs=CONSTS.RATE), nperseg=CONSTS.WINDOW_SIZE,
                 noverlap=CONSTS.OVERLAP, freq_axis=0)
