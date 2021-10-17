#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

from solution.data import SAMPLING_RATE, SAMPLES_LEN


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text



q = queue.Queue()


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
    global waveform
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

    return audio_line,



fig_upd_interval = 30
length = int(1.5 * SAMPLING_RATE)
waveform = np.zeros(length)

fig, ax = plt.subplots()
audio_line,  = ax.plot(waveform)


ax.axis((0, len(waveform), -1, 1))
ax.set_yticks([0])
ax.yaxis.grid(True)
ax.tick_params(bottom=False, top=False, labelbottom=False,
               right=False, left=False, labelleft=False)
fig.tight_layout(pad=0)

stream = sd.InputStream(channels=1, samplerate=SAMPLING_RATE, callback=audio_callback)
ani = FuncAnimation(fig, update_plot, interval=fig_upd_interval, blit=True)
with stream:
    plt.show()