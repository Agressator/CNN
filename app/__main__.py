import sys
import os
from argparse import ArgumentParser
from app.mnist_data import MNISTData
from app.neural_network import NeuralNetwork

MNIST_DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'mnist_data')


def create_args_parser(argv=sys.argv[1:]):
    parser = ArgumentParser()

    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1, help='')
    parser.add_argument('-bs', '--bath-size', type=int, default=1, help='')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='')
    parser.add_argument('--hidden-size', type=int, default=392, help="")

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = create_args_parser()

    mnist = MNISTData(MNIST_DATA_PATH)
    test_images, test_labels = mnist.get_test_data()
    train_images, train_labels = mnist.get_train_data()

    neural_network = NeuralNetwork(args.hidden_size)
    neural_network.train(train_images[0], train_labels[0])
    pass
