import torchvision
import random

trainset = torchvision.datasets.MNIST(root='../../../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../../../data/MNIST', train=False, download=True)


def gather_examples(dataset, in_filename, initiated_filename, holds_filename, digits_filename,
                    keeping_digits=(0, 1), start_sequence=(1, ), end_sequence=(0, )):
    digits = [
        (ident, digit)
        for ident, (image, digit) in enumerate(dataset)
        if digit in keeping_digits
    ]

    random.shuffle(digits)

    with open(in_filename, 'w') as in_f:
        with open(initiated_filename, 'w') as init_f:
            with open(holds_filename, 'w') as holds_f:
                with open(digits_filename, 'w') as digits_f:
                        last_digit = None
                        in_sequence = False

                        for t, (image, digit) in enumerate(digits):
                            digits_f.write('digit({}, {}).\n'.format(image, digit))

                            in_f.write('happensAt({}, {}).\n'.format(image, t))

                            holds_f.write(
                                'holdsAt(sequence = {}, {}).\n'.format(
                                    'true' if in_sequence else 'false', t
                                )
                            )

                            if last_digit in start_sequence and digit in start_sequence:
                                init_f.write('initiatedAt(sequence = true, {}).\n'.format(t))

                                in_sequence = True
                            elif last_digit in end_sequence and digit in end_sequence:
                                init_f.write('initiatedAt(sequence = false, {}).\n'.format(t))

                                in_sequence = False

                            last_digit = digit

                        in_f.write(
                            'allTimeStamps([{}]).\n'.format(
                                ', '.join(map(str, range(len(digits))))
                            )
                        )


def generate_data():
    gather_examples(
        trainset, 'in_train_data.txt', 'init_train_data.txt', 'holds_train_data.txt', 'digits_train_data.txt'
    )
    gather_examples(
        testset, 'in_test_data.txt', 'init_test_data.txt', 'holds_test_data.txt', 'digits_test_data.txt'
    )


if __name__ == '__main__':
    generate_data()
