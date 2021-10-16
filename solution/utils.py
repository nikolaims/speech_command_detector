import sounddevice

from solution.data import SAMPLING_RATE, SAMPLES_LEN
import torch
import numpy as np


def play(waveform, fs=SAMPLING_RATE):
    """
    Play waveform array with sampling frequency FS
    :param fs: sampling freq
    :param waveform: waveform numpy array
    """
    sounddevice.play(waveform, fs, blocking=True)


def apply_model(x, model, transform, hop_ms):
    p = np.zeros(len(x))
    hop = int(hop_ms/1000*SAMPLING_RATE)
    with torch.no_grad():
        for start in range(0, len(x)-SAMPLES_LEN, hop):
            x_slice = x[start:start+SAMPLES_LEN]
            prob = torch.sigmoid(model(transform(x_slice.reshape(1, -1)))).item()
            p[start:start + SAMPLES_LEN] += prob
    return p/(SAMPLES_LEN/hop)