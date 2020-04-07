import random

import numpy as np


def create_initated_at(digit, start_sequence, end_sequence, init_f, init_digit_f, in_sequence, t, is_true):
    if digit in start_sequence:
        seq_id = start_sequence.index(digit)

        init_f.write(
            'initiatedAt{}(sequence{} = true, {}).\n'.format(
                '' if is_true else 'Noise',
                seq_id,
                t
            )
        )
        init_digit_f.write(
            'initiatedAt{}(sequence{} = true, {}).\n'.format(
                '' if is_true else 'Noise',
                seq_id,
                t
            )
        )

        in_sequence[seq_id] = True
    elif digit in end_sequence:
        seq_id = end_sequence.index(digit)

        init_f.write(
            'initiatedAt{}(sequence{} = false, {}).\n'.format(
                '' if is_true else 'Noise',
                seq_id,
                t
            )
        )
        init_digit_f.write(
            'initiatedAt{}(sequence{} = false, {}).\n'.format(
                '' if is_true else 'Noise',
                seq_id,
                t
            )
        )

        in_sequence[seq_id] = False


def mnist_true_values(dataset):
    return [
        (ident, digit)
        for ident, (image, digit) in enumerate(dataset)
    ]


def gather_examples(dataset, in_filename, initiated_filename, holds_filename, network_filename, init_network_filename,
                    threshold, scenario_function, relevant_digits, start_sequence=None, end_sequence=None,
                    get_true_values=mnist_true_values, network_clause='digit'):
    if start_sequence is None:
        start_sequence = [0, 2, 4, 6, 8]
    if end_sequence is None:
        end_sequence = [1, 3, 5, 7, 9]

    available_digits = start_sequence + end_sequence

    true_values = get_true_values(dataset)

    random.shuffle(true_values)

    with open(in_filename, 'w') as in_f:
        with open(initiated_filename, 'w') as init_f:
            with open(holds_filename, 'w') as holds_f:
                with open(network_filename, 'w') as network_f:
                    with open(init_network_filename, 'w') as init_network_f:
                        last_network = []
                        in_sequence = [False] * 5

                        for t, (image, network) in enumerate(true_values):
                            network_f.write('{}({}, {}).\n'.format(network_clause, image, network))
                            init_network_f.write('{}({}, {}).\n'.format(network_clause, image, network))

                            in_f.write('happensAt({}, {}).\n'.format(image, t))

                            for i, s in enumerate(in_sequence):
                                holds_f.write(
                                    'holdsAt(sequence{} = {}, {}).\n'.format(
                                        i, 'true' if s else 'false', t
                                    )
                                )

                            if last_network:
                                digit_to_create, is_true = scenario_function(
                                    network, last_network, threshold, available_digits
                                )

                                if digit_to_create is not None:
                                    create_initated_at(
                                        digit_to_create, start_sequence, end_sequence,
                                        init_f, init_network_f, in_sequence, t, is_true
                                    )

                            last_network.append(network)
                            last_network = last_network[-relevant_digits:]

                        in_f.write(
                            'allTimeStamps([{}]).\n'.format(
                                ', '.join(map(str, range(len(true_values))))
                            )
                        )


def get_digit_for_initiated_at(digit, last_digits, threshold, available_digits):
    if digit == last_digits[-1]:
        return digit, True
    elif random.random() < threshold:
        return digit, False

    return None, None


def get_correct_digit_for_initiated_at(digit, last_digits, threshold, available_digits):
    if digit == last_digits[-1]:
        return digit, True

    return None, None


def default_noises(min_noise=0.0, max_noise=1.1, noise_step=0.2):
    return np.arange(min_noise, max_noise, noise_step)


def default_folder_name(noise):
    return 'noise_{0:.2f}/'.format(noise).replace('.', '_')
