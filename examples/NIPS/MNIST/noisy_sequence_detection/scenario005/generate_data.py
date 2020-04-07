import random
import numpy as np

from examples.NIPS.MNIST.noisy_sequence_detection.scenario001.generate_data import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.scenario004.generate_data import get_random_assignment


def get_digit_for_initiated_at(digit, last_digits, threshold, available_digits, assignment):
    if digit == last_digits[-1]:
        if random.random() < threshold:
            # True label gets swapped for a random one
            return random.choice(available_digits), False
        elif random.random() < threshold:
            # True label gets swapped for a pre-defined one
            return assignment[digit], False
        else:
            # True label is used
            return digit, True
    elif random.random() < threshold:
        # Random label is added
        return random.choice(available_digits), False

    return None, None


def noises(min_noise=0.0, max_noise=1.1, noise_step=0.2):
    return np.arange(min_noise, max_noise, noise_step)


def folder_name(noise):
    return 'noise_{0:.2f}/'.format(noise).replace('.', '_')


if __name__ == '__main__':
    assignment = get_random_assignment(list(range(10)))

    with open('assignment.txt', 'w') as o:
        o.write(str(assignment))

    generate_data(
        noises_function=noises,
        folder_name=folder_name,
        scenario_function=lambda *args, **kwargs: get_digit_for_initiated_at(
            *args, **kwargs, assignment=assignment
        )
    )
