import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario1.generate_data import generate_data


def get_digit_for_initiated_at(digit, last_digit, threshold, available_digits):
    if digit == last_digit:
        if random.random() < threshold:
            return random.choice(available_digits), False
        else:
            return digit, True

    return None, None


if __name__ == '__main__':
    generate_data(scenario_function=get_digit_for_initiated_at)
