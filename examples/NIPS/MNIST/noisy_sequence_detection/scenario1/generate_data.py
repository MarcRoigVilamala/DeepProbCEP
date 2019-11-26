import os
import numpy as np

import torchvision
import random

trainset = torchvision.datasets.MNIST(root='../../../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../../../data/MNIST', train=False, download=True)


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


def gather_examples(dataset, in_filename, initiated_filename, holds_filename, digits_filename, init_digit_filename,
                    threshold, scenario_function, start_sequence=None, end_sequence=None):
    if start_sequence is None:
        start_sequence = [0, 2, 4, 6, 8]
    if end_sequence is None:
        end_sequence = [1, 3, 5, 7, 9]

    available_digits = start_sequence + end_sequence

    digits = [
        (ident, digit)
        for ident, (image, digit) in enumerate(dataset)
    ]

    random.shuffle(digits)

    with open(in_filename, 'w') as in_f:
        with open(initiated_filename, 'w') as init_f:
            with open(holds_filename, 'w') as holds_f:
                with open(digits_filename, 'w') as digits_f:
                    with open(init_digit_filename, 'w') as init_digit_f:
                        last_digit = None
                        in_sequence = [False] * 5

                        for t, (image, digit) in enumerate(digits):
                            digits_f.write('digit({}, {}).\n'.format(image, digit))
                            init_digit_f.write('digit({}, {}).\n'.format(image, digit))

                            in_f.write('happensAt({}, {}).\n'.format(image, t))

                            for i, s in enumerate(in_sequence):
                                holds_f.write(
                                    'holdsAt(sequence{} = {}, {}).\n'.format(
                                        i, 'true' if s else 'false', t
                                    )
                                )

                            digit_to_create, is_true = scenario_function(
                                digit, last_digit, threshold, available_digits
                            )
                            if digit_to_create is not None:
                                create_initated_at(
                                    digit_to_create, start_sequence, end_sequence,
                                    init_f, init_digit_f, in_sequence, t, is_true
                                )

                            last_digit = digit

                        in_f.write(
                            'allTimeStamps([{}]).\n'.format(
                                ', '.join(map(str, range(len(digits))))
                            )
                        )


def get_digit_for_initiated_at(digit, last_digit, threshold, available_digits):
    if digit == last_digit:
        return digit, True
    elif random.random() < threshold:
        return digit, False

    return None, None


def get_correct_digit_for_initiated_at(digit, last_digit, threshold, available_digits):
    if digit == last_digit:
        return digit, True

    return None, None


def default_noises(min_noise=0.0, max_noise=1.1, noise_step=0.2):
    return np.arange(min_noise, max_noise, noise_step)


def default_folder_name(noise):
    return 'noise_{0:.2f}/'.format(noise).replace('.', '_')


def generate_data(noises_function=default_noises, folder_name=default_folder_name,
                  scenario_function=get_digit_for_initiated_at):
    in_train_data = 'in_train_data.txt'
    init_train_data = 'init_train_data.txt'
    holds_train_data = 'holds_train_data.txt'
    digits_train_data = 'digits_train_data.txt'
    init_digit_train_data = 'init_digit_train_data.txt'
    in_test_data = 'in_test_data.txt'
    init_test_data = 'init_test_data.txt'
    holds_test_data = 'holds_test_data.txt'
    digits_test_data = 'digits_test_data.txt'
    init_digit_test_data = 'init_digit_test_data.txt'

    for noise in noises_function():
        folder = folder_name(noise)

        if not os.path.exists(folder):
            os.makedirs(folder)

        iteration_in_train_data = folder + in_train_data
        iteration_init_train_data = folder + init_train_data
        iteration_holds_train_data = folder + holds_train_data
        iteration_digits_train_data = folder + digits_train_data
        iteration_init_digit_train_data = folder + init_digit_train_data
        iteration_in_test_data = folder + in_test_data
        iteration_init_test_data = folder + init_test_data
        iteration_holds_test_data = folder + holds_test_data
        iteration_digits_test_data = folder + digits_test_data
        iteration_init_digit_test_data = folder + init_digit_test_data

        gather_examples(
            trainset,
            iteration_in_train_data,
            iteration_init_train_data,
            iteration_holds_train_data,
            iteration_digits_train_data,
            iteration_init_digit_train_data,
            noise,
            scenario_function
        )
        gather_examples(
            testset,
            iteration_in_test_data,
            iteration_init_test_data,
            iteration_holds_test_data,
            iteration_digits_test_data,
            iteration_init_digit_test_data,
            0.0,
            get_correct_digit_for_initiated_at
        )


if __name__ == '__main__':
    generate_data()
