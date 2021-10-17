import sounddevice
import soundfile
from scipy.signal import resample

from solution.data import SAMPLING_RATE


def play(waveform, fs=SAMPLING_RATE):
    """
    Play waveform array with sampling frequency FS
    :param fs: sampling freq
    :param waveform: waveform numpy array
    """
    sounddevice.play(waveform, fs, blocking=True)

def rec(seconds):
    print('ON AIR ...')
    x = sounddevice.rec(int(seconds * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, blocking=True)
    print('DONE')
    return x

def load_audio_and_resample(path, target_fs=SAMPLING_RATE):
    x, fs = soundfile.read(path)
    if x.ndim == 2:
        x = x[:, 0]
    if fs == target_fs:
        return x
    return resample(x, int(len(x)*target_fs/fs))