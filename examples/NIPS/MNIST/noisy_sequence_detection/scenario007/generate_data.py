import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario001.generate_data import generate_data


def get_digit_for_initiated_at(digit, last_digits, threshold, available_digits, assignment):
    if digit == last_digits[-1]:
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
        )
    )
