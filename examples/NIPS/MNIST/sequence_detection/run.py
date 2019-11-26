import sys

from examples.NIPS.MNIST.mnist import MNIST_Net, test_MNIST, neural_predicate

sys.path.append('../../../')
from train import train_model, train, batch_train_model, train_batch
from test_utils import get_confusion_matrix, calculate_f1
from data_loader import load
from model import Model, Var
from optimizer import Optimizer
from network import Network
import torch

CONFUSION_INDEX = {
    'happensAt': {
        'nothing': 0,
        'something': 1
    },
    'initiatedAt': {
        'false': 0,
        'true': 1
    }
}


def get_target_happens_at(query):
    args0, args1, args2 = query.args

    return str(args1)


def query_transformation_happens_at(query):
    args0, _, args2 = query.args

    return query(args0, Var('X'), args2)


def get_result_happens_at(output):
    k, v = list(output.items())[0]

    result = str(k.args[1])
    prob = v[0]

    return result


def get_target_initiated_at(query):
    target = query.args[0].args[1]

    return str(target)


def query_transformation_initiated_at(query):
    args0, args1 = query.args

    args0args0 = args0.args[0]

    return query(args0(args0args0, Var('X')), args1)


def get_result_initiated_at(output):
    k, v = list(output.items())[0]

    result = str(k.args[0].args[1])
    prob = v[0]

    return result


TEST_METHODS = {
    'happensAt': (get_target_happens_at, query_transformation_happens_at, get_result_happens_at),
    'initiatedAt': (get_target_initiated_at, query_transformation_initiated_at, get_result_initiated_at)
}


def test(model, test_queries, test_functions=None):
    # Substitute the functions to the test ones
    original_functions = {
        net_name: model.networks[net_name].function
        for net_name in model.networks
    }
    if test_functions:
        for net_name, new_func in test_functions.items():
            model.networks[net_name].function = new_func

    with torch.no_grad():
        confusion_dict = get_confusion_matrix(
            model, CONFUSION_INDEX, test_queries, TEST_METHODS
        )

    f1_list = []
    for k, confusion in confusion_dict.items():
        f1 = calculate_f1(confusion)

        print(k)
        print(confusion)

        text = 'F1 {}: {}'.format(k, f1)
        print(text)

        f1_list.append(
            ('F1 {}'.format(k), f1)
        )

    # Restore the original function
    for net_name, ori_func in original_functions.items():
        model.networks[net_name].function = ori_func

    return f1_list


def run(training_data, test_data, problog_files):
    queries = load(training_data)
    test_queries = load(test_data)

    problog_string = ''
    for problog_file in problog_files:
        with open(problog_file) as f:
            problog_string += f.read()
            problog_string += '\n\n'

    network = MNIST_Net()
    net = Network(network, 'mnist_net', neural_predicate)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    model = Model(problog_string, [net], caching=False)
    optimizer = Optimizer(model, 2)

    train_model(model, queries, 1, optimizer, test_iter=1000, test=test_MNIST, snapshot_iter=10000)


if __name__ == '__main__':
    run(
        'holds_train_data.txt',
        'init_test_data.txt',
        ['ProbLogFiles/prob_ec_cached.pl', 'ProbLogFiles/event_defs.pl', 'in_train_data.txt']
    )
