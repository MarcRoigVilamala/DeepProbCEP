import json
import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario100_2.generate_data import generate_data, \
    get_correct_digit_for_initiated_at
from examples.NIPS.UrbanSounds8K.SequenceDetection.generate_data import sound_true_values
import data as data_module


if __name__ == '__main__':
    config = json.load(open('../../my-config_generate.json'))

    data_manager = getattr(data_module, config['data']['type'])(config['data'])

    t_loader = data_manager.get_loader('train', transfs=None)
    v_loader = data_manager.get_loader('val', transfs=None)

    generate_data(
        noises_function=lambda: [0.0],
        scenario_function=get_correct_digit_for_initiated_at,
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=4,
        training_set=t_loader,
        testing_set=v_loader,
        start_sequence=['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren'],
        end_sequence=['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music'],
        get_true_values=sound_true_values,
        network_clause='sound',
    )
