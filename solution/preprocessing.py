import numpy as np
import torch

from scipy import signal
from solution.data import SCDataset, SAMPLING_RATE, SAMPLES_LEN
from torchvision.transforms import Compose



class Spectrogram:
    def __init__(self, n_fft=512+1, hop_length=None, fs=SAMPLING_RATE, samples_len=SAMPLES_LEN):
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2 + 1

        self.freqs = np.arange(0, self.n_fft // 2 + 1) / n_fft * fs
        self.times = np.arange(self.hop_length, samples_len-self.hop_length, self.hop_length) / self.fs

    def __call__(self, waveform):
        _f, _t, spec = signal.spectrogram(waveform, self.fs, nperseg=self.n_fft, nfft=self.n_fft,
                                          noverlap=self.n_fft-self.hop_length, scaling='spectrum')
        return spec


class NormalizeSpec:
    def __call__(self, spec):
        spec = np.log10(spec + 1e-9)
        spec = (spec - spec.mean())/spec.std()
        return spec


class ToTensor:
    def __call__(self, spec):
        return torch.from_numpy(spec).to(torch.float).unsqueeze(0)


class NumberToTensor:
    def __call__(self, number):
        return torch.tensor(number).to(torch.float).unsqueeze(0)


class MainTransform(Compose):
    def __init__(self):
        super().__init__([Spectrogram(), NormalizeSpec(), ToTensor()])

if __name__ == '__main__':
    import pylab as plt


    from solution.utils import play

    dataset = SCDataset(r'/Users/kolai/Data/speech_commands_v0.01/ref_small_1000.csv')

    n = 9
    waveform, label = dataset[n]
    play(waveform.T)

    spectrogram = Spectrogram()
    transform = Compose([spectrogram, NormalizeSpec()])
    s = transform(waveform)

    f = spectrogram.freqs
    t = spectrogram.times
    _fig, axes = plt.subplots(2)
    axes[0].plot(t, s.mean(0))
    axes[0].axhline(s.mean(0).mean())
    axes[1].pcolormesh(t, f, s, cmap='seismic')
    plt.show()

    s_list = []
    for waveform, label in dataset:
        s = transform(waveform)
        s_list.append(s)

    plt.figure()
    plt.hist(np.array(s_list).std(1).std(1), bins=50)