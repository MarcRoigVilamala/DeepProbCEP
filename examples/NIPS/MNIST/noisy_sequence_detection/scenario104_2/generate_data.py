import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario001.generate_data import generate_data
from examples.NIPS.MNIST.noisy_sequence_detection.scenario004.generate_data import get_random_assignment
from examples.NIPS.MNIST.noisy_sequence_detection.scenario103_2.generate_data import get_correct_digit_for_initiated_at


def get_initiated_at_scenario104(digit, last_digits, threshold, available_digits, assignment):
    if digit in last_digits:
        if random.random() < threshold:
            return assignment[available_digits.index(digit)], False
        else:
            return digit, True

    return None, None


if __name__ == '__main__':
    assignment = get_random_assignment(list(range(10)))

    with open('assignment.txt', 'w') as o:
        o.write(str(assignment))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_initiated_at_scenario104(
            *args, **kwargs, assignment=assignment
        ),
        test_function=get_correct_digit_for_initiated_at,
        relevant_digits=1
    )
