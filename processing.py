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
    time_shift = 2 * np.pi * CONSTS.HOP_SIZE / CONSTS.WINDOW_SIZE * np.arange(dt_phase.shape[-1])
    dt_phase = (dt_phase.T + time_shift).T
    dk_phase = dk_phase + np.pi

    # recenter to (-pi, pi)
    dt_phase = np.mod(dt_phase + np.pi, 2 * np.pi) - np.pi
    dk_phase = np.mod(dk_phase + np.pi, 2 * np.pi) - np.pi

    return phase, dt_phase, dk_phase


def preprocess(audio_track):
    amplitude, phase = fourierTransform(audio_track)
    return (amplitude,) + differentiateAndRescale(phase)


def make_train_input(tracks, context_length):
    amplitude_out = list()
    phase_out = list()
    for track in tracks:
        amplitude, phase, dt_phase, dk_phase = preprocess(track.audio)
        i = context_length
        while i + context_length < dt_phase.shape[-1]:
            amplitude_out.append(amplitude[:, :, i - context_length:i + context_length])

            dt_phase_out.append(dk[:, :, i - context_length:i + context_length])
            dk_phase_out.append(amplitude[:, :, i - context_length:i + context_length])
    return amplitude_out, dt_phase_out, dk_phase_out


def make_train_data(tracks, context_length, target):
    amplitude_out = list()
    dt_phase_out = list()
    dk_phase_out = list()
    for track in tracks:
        amplitude, phase, dt_phase, dk_phase = preprocess(track.targets[target].audio)
        i = context_length
        while i + context_length < dt_phase.shape[-1]:
            amplitude_out.append(amplitude[:, :, i - context_length:i + context_length])
            dt_phase_out.append(amplitude[:, :, i - context_length:i + context_length])
            dk_phase_out.append(amplitude[:, :, i - context_length:i + context_length])
    return amplitude_out, dt_phase_out, dk_phase_out


def reconstruct(out_amplitude, out_phase):
    """reconstruct output from predicted amplitude and phase"""
    return istft(out_amplitude * np.exp(1j * out_phase,fs=CONSTS.RATE), nperseg=CONSTS.WINDOW_SIZE,
                 noverlap=CONSTS.OVERLAP, freq_axis=0)
