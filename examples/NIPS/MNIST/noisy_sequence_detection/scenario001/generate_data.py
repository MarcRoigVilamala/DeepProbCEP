import os

import torchvision
from examples.NIPS.generate_data_utils import mnist_true_values, gather_examples, get_digit_for_initiated_at, \
    get_correct_digit_for_initiated_at, default_noises, default_folder_name

trainset = torchvision.datasets.MNIST(root='../../../../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../../../../data/MNIST', train=False, download=True)


def generate_data(noises_function=default_noises, folder_name=default_folder_name,
                  scenario_function=get_digit_for_initiated_at, test_function=get_correct_digit_for_initiated_at,
                  relevant_digits=1, training_set=trainset, testing_set=testset, start_sequence=None, end_sequence=None,
                  get_true_values=mnist_true_values, network_clause='digit'):
    in_train_data = 'in_train_data.txt'
    init_train_data = 'init_train_data.txt'
    holds_train_data = 'holds_train_data.txt'
    digits_train_data = '{}s_train_data.txt'.format(network_clause)
    init_digit_train_data = 'init_{}_train_data.txt'.format(network_clause)
    in_test_data = 'in_test_data.txt'
    init_test_data = 'init_test_data.txt'
    holds_test_data = 'holds_test_data.txt'
    digits_test_data = '{}s_test_data.txt'.format(network_clause)
    init_digit_test_data = 'init_{}_test_data.txt'.format(network_clause)

    for noise in noises_function():
        folder = folder_name(noise)

        if not os.path.exists(folder):
            os.makedirs(folder)

        iteration_in_train_data = folder + in_train_data
        iteration_init_train_data = folder + init_train_data
        iteration_holds_train_data = folder + holds_train_data
        iteration_digits_train_data = folder + digits_train_data
        iteration_init_digit_train_data = folder + init_digit_train_data
        iteration_in_test_data = folder + in_test_data
        iteration_init_test_data = folder + init_test_data
        iteration_holds_test_data = folder + holds_test_data
        iteration_digits_test_data = folder + digits_test_data
        iteration_init_digit_test_data = folder + init_digit_test_data

        gather_examples(
            dataset=training_set,
            in_filename=iteration_in_train_data,
            initiated_filename=iteration_init_train_data,
            holds_filename=iteration_holds_train_data,
            network_filename=iteration_digits_train_data,
            init_network_filename=iteration_init_digit_train_data,
            threshold=noise,
            scenario_function=scenario_function,
            relevant_digits=relevant_digits,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            get_true_values=get_true_values,
            network_clause=network_clause
        )
        gather_examples(
            dataset=testing_set,
            in_filename=iteration_in_test_data,
            initiated_filename=iteration_init_test_data,
            holds_filename=iteration_holds_test_data,
            network_filename=iteration_digits_test_data,
            init_network_filename=iteration_init_digit_test_data,
            threshold=0.0,
            scenario_function=test_function,
            relevant_digits=relevant_digits,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            get_true_values=get_true_values,
            network_clause=network_clause
        )


if __name__ == '__main__':
    generate_data()
