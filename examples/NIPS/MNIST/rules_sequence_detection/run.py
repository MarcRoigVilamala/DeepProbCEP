import os
import sys

from examples.NIPS.ActivityDetection.prob_ec_testing import test
from examples.NIPS.MNIST.mnist import MNIST_Net, test_MNIST, neural_predicate
from examples.NIPS.MNIST.noisy_sequence_detection.run import run

if __name__ == '__main__':
    for folder in sorted(os.listdir(os.curdir)):
        if folder.startswith('rules'):
            print("#######################################################################################")
            print(folder)

            run(
                '../complex_sequence_detection/init_train_data.txt',
                '../complex_sequence_detection/init_digit_test_data.txt',
                [
                    'ProbLogFiles/prob_ec_cached.pl'
                ],
                problog_train_files=[
                    '../complex_sequence_detection/in_train_data.txt',
                    '{}/event_defs.pl'.format(folder)
                ],
                problog_test_files=[
                    '../complex_sequence_detection/in_test_data.txt',
                    'ProbLogFiles/event_defs.pl'
                ]
            )
