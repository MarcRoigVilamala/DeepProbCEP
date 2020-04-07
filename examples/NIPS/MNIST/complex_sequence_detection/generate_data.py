import torchvision
import random

from examples.NIPS import gather_examples as general_gather_examples
from examples.NIPS.generate_data_utils import mnist_true_values

trainset = torchvision.datasets.MNIST(root='../../../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../../../data/MNIST', train=False, download=True)


def gather_examples(dataset, in_filename, initiated_filename, holds_filename, network_filename, init_network_filename,
                    start_sequence=None, end_sequence=None, get_true_values=mnist_true_values, network_clause='digit'):
    threshold = None

    def scenario_function(digit, last_digits, threshold, available_digits):
        if digit == last_digits[-1]:
            return digit, True

        return None, None

    relevant_digits = 1

    general_gather_examples(
        dataset, in_filename, initiated_filename, holds_filename, network_filename, init_network_filename,
        threshold, scenario_function, relevant_digits, start_sequence, end_sequence, get_true_values, network_clause
    )


def generate_data():
    gather_examples(
        trainset, 'in_train_data.txt', 'init_train_data.txt', 'holds_train_data.txt', 'digits_train_data.txt',
        'init_digit_train_data.txt'
    )
    gather_examples(
        testset, 'in_test_data.txt', 'init_test_data.txt', 'holds_test_data.txt', 'digits_test_data.txt',
        'init_digit_test_data.txt'
    )


if __name__ == '__main__':
    generate_data()
