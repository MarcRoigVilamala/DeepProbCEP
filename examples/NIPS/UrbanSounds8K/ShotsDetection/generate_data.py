import json
import random

import torchvision
import data as data_module
from examples.NIPS.MNIST.complex_sequence_detection.generate_data import gather_examples
import pandas as pd


preprocessable = pd.read_pickle('/home/roigvilamalam/projects/Urban-Sound-Classification/preprocessable.pkl')
preprocessable = preprocessable[preprocessable['preprocessable']]

preprocessable_filenames = set(preprocessable['filename'])


def shots_true_values(dataset, keep_classes=('air_conditioner', )):
    return [
        (
            "'examples/NIPS/UrbanSounds8K{}'".format(sample['path'][2:]),
            'start' if sample['class'] in keep_classes else 'other'
        )
        for sample in dataset.dataset.data_arr
        if sample['path'][3:] in preprocessable_filenames
    ]


def gather_examples_shots(dataset, in_filename, initiated_filename, holds_filename, network_filename,
                          init_network_filename, start_sequence=None, end_sequence=None,
                          get_true_values=shots_true_values, network_clause='digit'):
    if start_sequence is None:
        start_sequence = ['start']
    if end_sequence is None:
        end_sequence = ['other']

    true_values = get_true_values(dataset)

    random.shuffle(true_values)

    true_values = [
        (image, network)
        for (image, network) in true_values
        if network not in end_sequence or random.random() < 0.15
    ]

    with open(in_filename, 'w') as in_f:
        with open(initiated_filename, 'w') as init_f:
            with open(holds_filename, 'w') as holds_f:
                with open(network_filename, 'w') as network_f:
                    with open(init_network_filename, 'w') as init_network_f:
                        last_network = None
                        in_sequence = False

                        timestamps = []

                        for t, (image, network) in enumerate(true_values):
                            timestamps.append(t)

                            network_f.write('{}({}, {}).\n'.format(network_clause, image, network))
                            init_network_f.write('{}({}, {}).\n'.format(network_clause, image, network))

                            in_f.write('happensAt({}, {}).\n'.format(image, t))

                            holds_f.write(
                                'holdsAt(sequence0 = {}, {}).\n'.format(
                                    'true' if in_sequence else 'false', t
                                )
                            )

                            if network in start_sequence and last_network in start_sequence:
                                init_f.write('initiatedAt(sequence0 = true, {}).\n'.format(t))
                                init_network_f.write('initiatedAt(sequence0 = true, {}).\n'.format(t))

                                in_sequence = True
                            elif network not in start_sequence and last_network not in start_sequence:
                                init_f.write('initiatedAt(sequence0 = false, {}).\n'.format(t))
                                init_network_f.write('initiatedAt(sequence0 = false, {}).\n'.format(t))

                                in_sequence = False

                            last_network = network

                        in_f.write(
                            'allTimeStamps([{}]).\n'.format(
                                ', '.join(map(str, timestamps))
                            )
                        )


def generate_data():
    config = json.load(open('../my-config_generate.json'))

    data_manager = getattr(data_module, config['data']['type'])(config['data'])

    t_loader = data_manager.get_loader('train', transfs=None)
    v_loader = data_manager.get_loader('val', transfs=None)

    gather_examples_shots(
        t_loader, 'in_train_data.txt', 'init_train_data.txt', 'holds_train_data.txt', 'sounds_train_data.txt',
        'init_sound_train_data.txt', get_true_values=shots_true_values, network_clause='shots',
        start_sequence=['start'],
        end_sequence=['other']
    )
    gather_examples_shots(
        v_loader, 'in_test_data.txt', 'init_test_data.txt', 'holds_test_data.txt', 'sounds_test_data.txt',
        'init_sound_test_data.txt', get_true_values=shots_true_values, network_clause='shots',
        start_sequence=['start'],
        end_sequence=['other']
    )


if __name__ == '__main__':
    generate_data()
