import json

import torchvision
import data as data_module
from examples.NIPS.generate_data_utils import gather_examples
import pandas as pd


preprocessable = pd.read_pickle('/home/roigvilamalam/projects/Urban-Sound-Classification/preprocessable.pkl')
preprocessable = preprocessable[preprocessable['preprocessable']]

preprocessable_filenames = set(preprocessable['filename'])


def sound_true_values(dataset):
    return [
        (
            "'examples/NIPS/UrbanSounds8K{}'".format(sample['path'][2:]),
            sample['class']
        )
        for sample in dataset.dataset.data_arr
        if sample['path'][3:] in preprocessable_filenames
    ]


def generate_data():
    config = json.load(open('../my-config_generate.json'))

    data_manager = getattr(data_module, config['data']['type'])(config['data'])

    t_loader = data_manager.get_loader('train', transfs=None)
    v_loader = data_manager.get_loader('val', transfs=None)

    def scenario_function(digit, last_digits, threshold, available_digits):
        if digit == last_digits[-1]:
            return digit, True

        return None, None

    gather_examples(
        t_loader, 'in_train_data.txt', 'init_train_data.txt', 'holds_train_data.txt', 'sounds_train_data.txt',
        'init_sound_train_data.txt', get_true_values=sound_true_values, network_clause='sound',
        start_sequence=['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren'],
        end_sequence=['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music'],
        relevant_digits=1, scenario_function=scenario_function, threshold=None
    )
    gather_examples(
        v_loader, 'in_test_data.txt', 'init_test_data.txt', 'holds_test_data.txt', 'sounds_test_data.txt',
        'init_sound_test_data.txt', get_true_values=sound_true_values, network_clause='sound',
        start_sequence=['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren'],
        end_sequence=['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music'],
        relevant_digits=1, scenario_function=scenario_function, threshold=None
    )


if __name__ == '__main__':
    generate_data()
