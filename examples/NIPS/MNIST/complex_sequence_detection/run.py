import sys

from examples.NIPS.ActivityDetection.prob_ec_testing import test
from examples.NIPS.MNIST.mnist import MNIST_Net, test_MNIST, neural_predicate

sys.path.append('../../../')
from train import train_model, train, batch_train_model, train_batch
from test_utils import get_confusion_matrix, calculate_f1
from data_loader import load
from model import Model, Var
from optimizer import Optimizer
from network import Network
import torch


def add_files_to(problog_files, problog_string):
    for problog_file in problog_files:
        with open(problog_file) as f:
            problog_string += f.read()
            problog_string += '\n\n'

    return problog_string


def my_test(model_to_test, test_queries, test_functions=None):
    res = test(model_to_test, test_queries, test_functions=test_functions)

    # res += test_MNIST(model_to_test)

    return res


def run(training_data, test_data, problog_files, problog_train_files=(), problog_test_files=()):
    queries = load(training_data)
    test_queries = load(test_data)

    problog_string = add_files_to(problog_files, '')

    problog_train_string = add_files_to(problog_train_files, problog_string)
    problog_test_string = add_files_to(problog_test_files, problog_string)

    network = MNIST_Net()
    net = Network(network, 'mnist_net', neural_predicate)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    model_to_train = Model(problog_train_string, [net], caching=False)
    optimizer = Optimizer(model_to_train, 2)

    model_to_test = Model(problog_test_string, [net], caching=False)

    train_model(
        model_to_train,
        queries,
        5,
        optimizer,
        test_iter=1000,
        test=lambda _: my_test(
            model_to_test,
            test_queries,
            test_functions={
                'mnist_net': lambda *args, **kwargs: neural_predicate(
                    *args, **kwargs, dataset='test'
                )
            },
        ),
        snapshot_iter=len(queries)
    )


if __name__ == '__main__':
    run(
        'init_train_data.txt',
        'holds_test_data.txt',
        ['ProbLogFiles/prob_ec_cached.pl', 'ProbLogFiles/event_defs.pl'],
        problog_train_files=['in_train_data.txt'],
        problog_test_files=['in_test_data.txt']
    )
