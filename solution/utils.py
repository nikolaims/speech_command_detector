import sounddevice

from solution.data import SAMPLING_RATE


def play(waveform, fs=SAMPLING_RATE):
    """
    Play waveform array with sampling frequency FS
    :param fs: sampling freq
    :param waveform: waveform numpy array
    """
    sounddevice.play(waveform, fs, blocking=True)
