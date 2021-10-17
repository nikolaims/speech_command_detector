import numpy as np
import pylab as plt

from solution.data import SAMPLING_RATE


def plot_offline(x, p, title):

    t = np.arange(len(x))/SAMPLING_RATE
    plt.plot(t, x, label='audio')
    plt.fill_between(t, -np.abs(x).max() * (p >= 0.5), np.abs(x).max() * (p >= 0.5),
                     alpha=0.5, color='C1', label='model selection')
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()