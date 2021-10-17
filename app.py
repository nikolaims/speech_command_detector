import argparse


def file_input_handle(args):
    print(args.path, '*** FILE SPOTTING')


def mic_input_handle(args):
    if args.record:
        print(args.record, '*** RECORD SPOTTING')
    else:
        print('*** REALTIME SPOTTING')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(required=True , title='Input type')

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
