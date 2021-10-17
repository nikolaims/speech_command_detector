# based on https://python-sounddevice.readthedocs.io/en/0.4.2/examples.html#plot-microphone-signal-s-in-real-time

import queue
import sys
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

from matplotlib.animation import FuncAnimation
from scipy.signal.windows import tukey
from solution.data import SAMPLING_RATE, SAMPLES_LEN


def realtime_mic_spotting(hop_ms=100, fig_upd_interval=30, length_sec=3):
    hop = int(hop_ms * SAMPLING_RATE / 1000)
    q = queue.Queue()
    window = tukey(SAMPLES_LEN, 0.75)**2

    from solution.model import ConvNet
    from solution.infer import InferModel
    model_name = 'small_1000'
    model_state_path = f'results/{model_name}.pt'
    infer_model = InferModel(ConvNet, model_state_path, out_format='proba')

    def audio_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        q.put(indata[:, 0].copy())

    def update_plot(frame):
        """This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.

        """
        nonlocal waveform, proba, weights
        total_shift = 0
        while True:
            try:
                data = q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            total_shift += shift
            waveform = np.roll(waveform, -shift)
            waveform[-shift:] = data
        audio_line.set_ydata(waveform)

        proba = np.roll(proba, -total_shift)
        proba[-total_shift:] = np.nan
        weights = np.roll(weights, -total_shift)
        weights[-total_shift:] = 0

        if np.all(np.isnan(proba[-hop:])):
            p_ = proba[-SAMPLES_LEN:]
            p_[np.isnan(p_)] = 0
            p = infer_model(waveform[-SAMPLES_LEN:])
            weights[-SAMPLES_LEN:] += window.copy()
            proba[-SAMPLES_LEN:] += window.copy()*p

        proba_line.set_ydata((proba / weights >= 0.5) * 1.)
        proba_line2.set_ydata(-proba_line.get_ydata())
        return audio_line, proba_line, proba_line2

    length = int(length_sec * SAMPLING_RATE)
    waveform = np.zeros(length)
    proba = np.zeros(length)*np.nan
    weights = np.zeros(length)

    fig, ax = plt.subplots()
    audio_line,  = ax.plot(waveform)
    proba_line,  = ax.plot(proba, color='C1')
    proba_line2,  = ax.plot(-proba, color='C1')
    ax.axis((0, len(waveform), -1.1, 1.1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.InputStream(channels=1, samplerate=SAMPLING_RATE, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=fig_upd_interval, blit=True)
    with stream:
        plt.show()


if __name__ == '__main__':
    realtime_mic_spotting()
