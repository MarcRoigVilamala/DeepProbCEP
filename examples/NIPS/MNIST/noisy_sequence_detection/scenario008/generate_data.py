import random

from examples.NIPS.MNIST.noisy_sequence_detection.scenario001.generate_data import generate_data


def get_digit_for_initiated_at(digit, last_digits, threshold, available_digits, swap):
    if digit == last_digits[-1]:
        if digit == swap[0] and random.random() < threshold:
            return swap[1], False
        else:
            return digit, True

    return None, None


if __name__ == '__main__':
    available_digits = list(range(10))

    swap_from = random.choice(available_digits)
    swap_to = random.choice(available_digits)
    # Ensure that we are not swapping a number for itself
    while swap_from == swap_to:
        swap_to = random.choice(available_digits)

    with open('swap.txt', 'w') as o:
        o.write('{},{}'.format(swap_from, swap_to))

    generate_data(
        scenario_function=lambda *args, **kwargs: get_digit_for_initiated_at(
            *args, **kwargs, swap=(swap_from, swap_to)
        )
    )
