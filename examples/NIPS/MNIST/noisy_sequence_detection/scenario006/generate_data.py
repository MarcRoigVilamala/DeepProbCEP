import random
import numpy as np

from examples.NIPS.MNIST.noisy_sequence_detection.scenario001.generate_data import generate_data, trainset
from examples.NIPS.MNIST.noisy_sequence_detection.scenario004.generate_data import get_random_assignment


DEFAULT_PROBABILITIES = {
    0: 0.1,
    1: 0.2,
    2: 0.3,
    3: 0.4,
    4: 0.5,
    5: 0.6,
    6: 0.7,
    7: 0.8,
    8: 0.9,
    9: 1.0,
}


def get_probability(t, probabilities=None):
    if probabilities is None:
        probabilities = DEFAULT_PROBABILITIES

    return probabilities[t[1]]


def filter_by_probability(t, probabilities=None):
    return random.random() < get_probability(t, probabilities)


if __name__ == '__main__':
    assignment = get_random_assignment(list(range(10)))

    with open('assignment.txt', 'w') as o:
        o.write(str(assignment))

    filtered_trainset = list(filter(filter_by_probability, trainset))

    generate_data(
        training_set=filtered_trainset,
    )
