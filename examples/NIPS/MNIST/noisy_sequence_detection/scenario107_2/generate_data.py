import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario001.generate_data import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.scenario103_2.generate_data import get_correct_digit_for_initiated_at


def get_digit_for_initiated_at(digit, last_digits, threshold, available_digits, assignment):
    if digit in last_digits:
        if digit == assignment and random.random() < threshold:
            return random.choice(available_digits), False
        else:
            return digit, True

    return None, None


if __name__ == '__main__':
    assignment = random.choice(list(range(10)))

    with open('assignment.txt', 'w') as o:
        o.write(str(assignment))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_digit_for_initiated_at(
            *args, **kwargs, assignment=assignment
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=1
    )
