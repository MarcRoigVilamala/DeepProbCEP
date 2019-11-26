import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario1.generate_data import generate_data


def get_digit_for_initiated_at(digit, last_digit, threshold, available_digits, assignment):
    if digit == last_digit:
        if random.random() < threshold:
            return assignment[digit], False
        else:
            return digit, True

    return None, None


def check_swap_for_itself(assignment):
    for i, a in enumerate(assignment):
        if i == a:
            return True

    return False


def get_random_assignment(digits):
    assignment = digits

    while check_swap_for_itself(assignment):
        random.shuffle(assignment)

    return assignment


if __name__ == '__main__':
    assignment = get_random_assignment(list(range(10)))

    with open('assignment.txt', 'w') as o:
        o.write(str(assignment))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_digit_for_initiated_at(
            *args, **kwargs, assignment=assignment
        )
    )
