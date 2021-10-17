import argparse

import sounddevice
import soundfile

from solution.data import SAMPLING_RATE
from solution.mic_rt import realtime_mic_spotting
from solution.viz import plot_offline
from solution.infer import spot_the_phrase


def file_input_handle(args):
    print(args.path, '*** FILE SPOTTING')
    x, fs = soundfile.read(args.path)
    assert fs == SAMPLING_RATE, f'Sample rate should be {SAMPLING_RATE}'

    p = spot_the_phrase(x)
    plot_offline(x, p, args.path)


def mic_input_handle(args):
    if args.record:
        print(args.record, 'SEC RECORDING SPOTTING')
        print('ON AIR ...')
        x = sounddevice.rec(int(args.record * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, blocking=True)
        print('DONE')
        p = spot_the_phrase(x)
        plot_offline(x, p, f'recording {args.record}s')
    else:
        print('*** REALTIME SPOTTING')
        realtime_mic_spotting()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(required=True, title='Input type')

    parser_file = subparsers.add_parser('file', help='audio file', description='open audio file and spot the phrase')
    parser_file.add_argument('path', help='audio file path to open and spot the phrase')
    parser_file.set_defaults(func=file_input_handle)

    parser_mic = subparsers.add_parser('mic', help='microphone',
                                       description='run real-time phrase spotting on microphone recording or '
                                                   '(if -r flag is given) record and then spot the phrase offline')
    parser_mic.add_argument('-r', '--record', type=int, help='record SEC seconds and spot the phrase offline',
                            metavar='SEC')
    parser_mic.set_defaults(func=mic_input_handle)

    args = parser.parse_args()
    args.func(args)
