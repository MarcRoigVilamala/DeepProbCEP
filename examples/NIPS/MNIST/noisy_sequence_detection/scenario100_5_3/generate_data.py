import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario001.generate_data import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.scenario100_3_3.generate_data import \
    get_correct_digit_for_initiated_at

if __name__ == '__main__':
    generate_data(
        scenario_function=lambda *args, **kwargs: get_correct_digit_for_initiated_at(*args, **kwargs, appearances=2),
        test_function=lambda *args, **kwargs: get_correct_digit_for_initiated_at(*args, **kwargs, appearances=2),
        relevant_digits=4,
        noises_function=lambda: [0.0]
    )
