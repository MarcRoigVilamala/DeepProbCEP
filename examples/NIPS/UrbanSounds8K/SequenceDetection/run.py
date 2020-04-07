import json

import sys

from examples.NIPS.UrbanSounds8K.sounds_utils import SoundsUtils, SoundLinearNet, neural_predicate_linear, SoundCNNet, \
    SoundVGGish, neural_predicate_vggish
from examples.NIPS.ActivityDetection.prob_ec_testing import test
from examples.NIPS.MNIST.complex_sequence_detection.run import add_files_to
from train import train_model
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
import torch


def my_test(model_to_test, test_queries, test_functions=None):
    res = test(model_to_test, test_queries, test_functions=test_functions)

    # res += test_MNIST(model_to_test)

    return res


def run(training_data, test_data, problog_files, problog_train_files=(), problog_test_files=(), config_file=None,
        net_mode='init', cfg=None):
    config = json.load(open(config_file))
    config['net_mode'] = net_mode
    config['cfg'] = cfg

    queries = load(training_data)
    test_queries = load(test_data)

    sounds = SoundsUtils(config)

    problog_string = add_files_to(problog_files, '')

    problog_train_string = add_files_to(problog_train_files, problog_string)
    problog_test_string = add_files_to(problog_test_files, problog_string)

    network = sounds.network
    net = Network(network, 'sound_net', sounds.neural_predicate)
    net.optimizer = sounds.optimizer
    model_to_train = Model(problog_train_string, [net], caching=False)
    optimizer = Optimizer(model_to_train, 2)

    model_to_test = Model(problog_test_string, [net], caching=False)

    train_model(
        model_to_train,
        queries,
        5,
        optimizer,
        test_iter=len(queries),
        test=lambda _: my_test(
            model_to_test,
            test_queries,
            test_functions={
                'sound_net': lambda *args, **kwargs: sounds.neural_predicate(
                    *args, **kwargs, in_training=False
                )
            },
        ),
        snapshot_iter=len(queries)
    )


def run_linear(training_data, test_data, problog_files, problog_train_files=(), problog_test_files=()):
    queries = load(training_data)
    test_queries = load(test_data)

    # network = SoundLinearNet()
    # network = SoundCNNet()
    network = SoundVGGish()

    problog_string = add_files_to(problog_files, '')

    problog_train_string = add_files_to(problog_train_files, problog_string)
    problog_test_string = add_files_to(problog_test_files, problog_string)

    net = Network(network, 'sound_net', neural_predicate_vggish)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    model_to_train = Model(problog_train_string, [net], caching=False)
    optimizer = Optimizer(model_to_train, 2)

    model_to_test = Model(problog_test_string, [net], caching=False)

    train_model(
        model_to_train,
        queries,
        nr_epochs=10,
        optimizer=optimizer,
        test_iter=len(queries),
        test=lambda _: my_test(
            model_to_test,
            test_queries
        ),
        log_iter=500,
        snapshot_iter=len(queries),
        snapshot_name=' SequenceDetectionSnapshots/model'
    )


if __name__ == '__main__':
    run_linear(
        # 'examples/NIPS/UrbanSounds8K/SequenceDetection/sounds_train_data_tiny.txt',
        # 'examples/NIPS/UrbanSounds8K/SequenceDetection/sounds_train_data_tiny.txt',
        'examples/NIPS/UrbanSounds8K/SequenceDetection/sounds_train_data.txt',
        'examples/NIPS/UrbanSounds8K/SequenceDetection/init_sound_test_data.txt',
        [
            'examples/NIPS/UrbanSounds8K/SequenceDetection/ProbLogFiles/prob_ec_cached.pl',
            'examples/NIPS/UrbanSounds8K/SequenceDetection/ProbLogFiles/event_defs.pl'
        ],
        problog_train_files=['examples/NIPS/UrbanSounds8K/SequenceDetection/in_train_data.txt'],
        problog_test_files=['examples/NIPS/UrbanSounds8K/SequenceDetection/in_test_data.txt']
    )
    # run(
    #     'examples/NIPS/UrbanSounds8K/SequenceDetection/sounds_train_data_tiny.txt',
    #     'examples/NIPS/UrbanSounds8K/SequenceDetection/sounds_train_data_tiny.txt',
    #     # 'examples/NIPS/UrbanSounds8K/SequenceDetection/sounds_train_data.txt',
    #     # 'examples/NIPS/UrbanSounds8K/SequenceDetection/init_sound_test_data.txt',
    #     [
    #         'examples/NIPS/UrbanSounds8K/SequenceDetection/ProbLogFiles/prob_ec_cached.pl',
    #         'examples/NIPS/UrbanSounds8K/SequenceDetection/ProbLogFiles/event_defs.pl'
    #     ],
    #     problog_train_files=['examples/NIPS/UrbanSounds8K/SequenceDetection/in_train_data.txt'],
    #     problog_test_files=['examples/NIPS/UrbanSounds8K/SequenceDetection/in_test_data.txt'],
    #     config_file='examples/NIPS/UrbanSounds8K/my-config.json',
    #     cfg='examples/NIPS/UrbanSounds8K/crnn.cfg'
    # )
