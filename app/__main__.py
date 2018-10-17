import sys
from argparse import ArgumentParser


def create_args_parser(argv=sys.argv[1:]):
    parser = ArgumentParser()

    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1, help='')
    parser.add_argument('-bs', '--bath-size', type=int, default=1, help='')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = create_args_parser()
