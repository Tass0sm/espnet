#!/usr/bin/env python3
from espnet2.tasks.utts import UTTSTask


def get_parser():
    parser = UTTSTask.get_parser()
    return parser


def main(cmd=None):
    """TTS training

    Example:

        % python utts_train.py ... TODO
    """
    UTTSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
