import argparse

from solution.mic_rt import realtime_mic_spotting
from solution.viz import plot_offline
from solution.infer import spot_the_phrase
from solution.utils import load_audio_and_resample, rec


def file_input_handle(args):
    print(args.path, '*** FILE SPOTTING')
    x = load_audio_and_resample(args.path)
    p = spot_the_phrase(x)
    plot_offline(x, p, args.path)


def mic_input_handle(args):
    if args.record:
        print(args.record, 'SEC RECORDING SPOTTING')
        x = rec(seconds=args.record)
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
