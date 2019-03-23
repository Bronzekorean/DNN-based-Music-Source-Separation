from consts import CONSTS
from scipy.signal import stft, istft, get_window
import numpy as np


def fourierTransform(audio_track):
    _, _, transformed = stft(audio_track, fs=CONSTS.RATE, nperseg=CONSTS.WINDOW_SIZE, noverlap=CONSTS.OVERLAP, axis=0)

    amplitude, phase = np.absolute(transformed), np.angle(transformed)
    return amplitude, phase


def differentiateAndRescale(phase):
    """Make data from audio recoding
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

    return phase, dt_phase, dk_phase


def preprocess(audio_track):
    amplitude, phase = fourierTransform(audio_track)
    return (amplitude,) + differentiateAndRescale(phase)


def extract_context(index, array, context_length=CONSTS.CONTEXT_SIZE):
    desired_shape = array.shape[:2]
    time_frames = array.shape[-1]
    if index < context_length:
        # pad with zeroes to fill context
        return np.concatenate((
            np.zeros((desired_shape + (context_length - index,))),
            array[:, :, 0:index + context_length + 1]), axis=-1)

    elif index + context_length < time_frames:
        return array[:, :, index - context_length:index + context_length + 1]

    else:
        return np.concatenate(
            [array[:, :, index - context_length:],
             np.zeros((desired_shape + (index + context_length - time_frames + 1 ,)))],
            axis=-1)


def make_train_input(tracks, context_length=CONSTS.CONTEXT_SIZE):
    amplitude_out = list()
    phase_out = list()
    for track in tracks:
        amplitude, _, dt_phase, dk_phase = preprocess(track.audio)
        time_frames = amplitude.shape[-1]
        amplitude_out = amplitude_out + [extract_context(i, amplitude, context_length) for i in range(time_frames)]
        phase_out = phase_out + [np.concatenate((extract_context(i, dt_phase, context_length),
                                                extract_context(i, dk_phase, context_length)), axis=1)
                                 for i in range(time_frames)]
    return [amplitude_out, phase_out]


def make_target_data(tracks, target, context_length=CONSTS.CONTEXT_SIZE):
    amplitude_out = None
    empty = True
    for track in tracks:
        amplitude, _, _, _ = preprocess(track.targets[target].audio)
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
