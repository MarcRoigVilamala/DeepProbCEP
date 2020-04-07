import json
import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario004.generate_data import get_random_assignment
from examples.NIPS.MNIST.noisy_sequence_detection.scenario100_2.generate_data import generate_data, \
    get_correct_digit_for_initiated_at
from examples.NIPS.MNIST.noisy_sequence_detection.scenario104_2.generate_data import get_initiated_at_scenario104
from examples.NIPS.UrbanSounds8K.SequenceDetection.generate_data import sound_true_values
import data as data_module


if __name__ == '__main__':
    config = json.load(open('../../my-config_generate.json'))

    data_manager = getattr(data_module, config['data']['type'])(config['data'])

    t_loader = data_manager.get_loader('train', transfs=None)
    v_loader = data_manager.get_loader('val', transfs=None)

    all_events = [
        'air_conditioner',
        'car_horn',
        'children_playing',
        'dog_bark',
        'drilling',
        'engine_idling',
        'gun_shot',
        'jackhammer',
        'siren',
        'street_music'
    ]

    start_sequence = ['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren']
    end_sequence = ['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music']

    numbers_assignment = get_random_assignment(list(range(10)))

    assignment = list(map(lambda x: all_events[x], numbers_assignment))

    with open('assignment.txt', 'w') as o:
        o.write(str(assignment))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_initiated_at_scenario104(
            *args, **kwargs, assignment=assignment
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=1,
        training_set=t_loader,
        testing_set=v_loader,
        start_sequence=start_sequence,
        end_sequence=end_sequence,
        get_true_values=sound_true_values,
        network_clause='sound',
    )
