import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario001.generate_data import generate_data
from itertools import islice


def contains_element_n_times(l, x, n):
    gen = (True for i in l if i == x)
    return next(islice(gen, n-1, None), False)


def get_correct_digit_for_initiated_at(digit, last_digits, threshold, available_digits, appearances):
    if contains_element_n_times(last_digits, digit, appearances):
        return digit, True

    return None, None


if __name__ == '__main__':
    generate_data(
        scenario_function=lambda *args, **kwargs: get_correct_digit_for_initiated_at(*args, **kwargs, appearances=2),
        test_function=lambda *args, **kwargs: get_correct_digit_for_initiated_at(*args, **kwargs, appearances=2),
        relevant_digits=2,
        noises_function=lambda: [0.0]
    )
