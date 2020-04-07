import json
import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario100_2.generate_data import generate_data, \
    get_correct_digit_for_initiated_at
from examples.NIPS.MNIST.noisy_sequence_detection.scenario108_2.generate_data import get_initiated_at_scenario108
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

    swap_from = random.choice(all_events)
    swap_to = random.choice(all_events)
    # Ensure that we are not swapping a number for itself
    while swap_from == swap_to:
        swap_to = random.choice(all_events)

    with open('swap.txt', 'w') as o:
        o.write('{},{}'.format(swap_from, swap_to))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_initiated_at_scenario108(
            *args, **kwargs, swap=(swap_from, swap_to)
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=1,
        training_set=t_loader,
        testing_set=v_loader,
        start_sequence=['air_conditioner', 'children_playing', 'drilling', 'gun_shot', 'siren'],
        end_sequence=['car_horn', 'dog_bark', 'engine_idling', 'jackhammer', 'street_music'],
        get_true_values=sound_true_values,
        network_clause='sound',
    )
