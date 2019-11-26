import torchvision
import random

trainset = torchvision.datasets.MNIST(root='../../../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../../../data/MNIST', train=False, download=True)


def gather_examples(dataset, in_filename, initiated_filename, holds_filename, digits_filename, init_digit_filename,
                    start_sequence=None, end_sequence=None):
    if start_sequence is None:
        start_sequence = [0, 2, 4, 6, 8]
    if end_sequence is None:
        end_sequence = [1, 3, 5, 7, 9]

    digits = [
        (ident, digit)
        for ident, (image, digit) in enumerate(dataset)
    ]

    random.shuffle(digits)

    with open(in_filename, 'w') as in_f:
        with open(initiated_filename, 'w') as init_f:
            with open(holds_filename, 'w') as holds_f:
                with open(digits_filename, 'w') as digits_f:
                    with open(init_digit_filename, 'w') as init_digit_f:
                        last_digit = None
                        in_sequence = [False] * 5

                        for t, (image, digit) in enumerate(digits):
                            digits_f.write('digit({}, {}).\n'.format(image, digit))
                            init_digit_f.write('digit({}, {}).\n'.format(image, digit))

                            in_f.write('happensAt({}, {}).\n'.format(image, t))

                            for i, s in enumerate(in_sequence):
                                holds_f.write(
                                    'holdsAt(sequence{} = {}, {}).\n'.format(
                                        i, 'true' if s else 'false', t
                                    )
                                )

                            if digit == last_digit:
                                if digit in start_sequence:
                                    seq_id = start_sequence.index(digit)

                                    init_f.write('initiatedAt(sequence{} = true, {}).\n'.format(seq_id, t))
                                    init_digit_f.write('initiatedAt(sequence{} = true, {}).\n'.format(seq_id, t))

                                    in_sequence[seq_id] = True
                                elif digit in end_sequence:
                                    seq_id = end_sequence.index(digit)

                                    init_f.write('initiatedAt(sequence{} = false, {}).\n'.format(seq_id, t))
                                    init_digit_f.write('initiatedAt(sequence{} = false, {}).\n'.format(seq_id, t))

                                    in_sequence[seq_id] = False

                            last_digit = digit

                        in_f.write(
                            'allTimeStamps([{}]).\n'.format(
                                ', '.join(map(str, range(len(digits))))
                            )
                        )


def generate_data():
    gather_examples(
        trainset, 'in_train_data.txt', 'init_train_data.txt', 'holds_train_data.txt', 'digits_train_data.txt',
        'init_digit_train_data.txt'
    )
    gather_examples(
        testset, 'in_test_data.txt', 'init_test_data.txt', 'holds_test_data.txt', 'digits_test_data.txt',
        'init_digit_test_data.txt'
    )


if __name__ == '__main__':
    generate_data()
