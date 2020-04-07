import torch
from problog.logic import Var

from test_utils import get_confusion_matrix, calculate_f1, calculate_accuracy

CONFUSION_INDEX = {
    'happensAt': {
        'nothing': 0,
        'something': 1
    },
    'initiatedAt': {
        'X=Y': 0,
        'sequence0=false': 1,
        'sequence1=false': 2,
        'sequence2=false': 3,
        'sequence3=false': 4,
        'sequence4=false': 5,
        'sequence0=true': 6,
        'sequence1=true': 7,
        'sequence2=true': 8,
        'sequence3=true': 9,
        'sequence4=true': 10,
   },
    'holdsAt': {
        'X': 0,
        'false': 1,
        'true': 2
    },
    'digit': {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
    },
    'sound': {
        'air_conditioner': 0,
        'car_horn': 1,
        'children_playing': 2,
        'dog_bark': 3,
        'drilling': 4,
        'engine_idling': 5,
        'gun_shot': 6,
        'jackhammer': 7,
        'siren': 8,
        'street_music': 9,
    },
    'shots': {
        'start': 0,
        'other': 1,
    }
}


def get_target_happens_at(query):
    args0, args1, args2 = query.args

    return str(args1)


def query_transformation_happens_at(query):
    args0, _, args2 = query.args

    return query(args0, Var('X'), args2)


def get_result_happens_at(output):
    k, v = list(output.items())[0]

    result = str(k.args[1])
    prob = v[0]

    return result


def get_target_initiated_at(query):
    target = query.args[0]

    return str(target)


def query_transformation_initiated_at(query):
    args0, args1 = query.args

    return query(args0(Var('X'), Var('Y')), args1)


def get_result_initiated_at(output):
    k, v = list(output.items())[0]

    result = str(k.args[0])
    prob = v[0]

    return result


def get_target_holds_at(query):
    target = query.args[0].args[1]

    return str(target)


def query_transformation_holds_at(query):
    args0, args1 = query.args

    args0args0 = args0.args[0]

    return query(args0(args0args0, Var('X')), args1)


def get_result_holds_at(output):
    k, v = list(output.items())[0]

    result = str(k.args[0].args[1])
    prob = v[0]

    return result


def get_target_digit(query):
    target = query.args[1]

    return str(target)


def query_transformation_digit(query):
    args0, _ = query.args

    return query(args0, Var('X'))


def get_result_digit(output):
    k, v = list(output.items())[0]

    result = str(k.args[1])
    prob = v[0]

    return result


TEST_METHODS = {
    'happensAt': (get_target_happens_at, query_transformation_happens_at, get_result_happens_at),
    'initiatedAt': (get_target_initiated_at, query_transformation_initiated_at, get_result_initiated_at),
    'holdsAt': (get_target_holds_at, query_transformation_holds_at, get_result_holds_at),
    'digit': (get_target_digit, query_transformation_digit, get_result_digit),
    'sound': (get_target_digit, query_transformation_digit, get_result_digit),
    'shots': (get_target_digit, query_transformation_digit, get_result_digit),
}


def test(model, test_queries, test_functions=None, confusion_index=None, test_methods=None):
    # Substitute the functions to the test ones
    if confusion_index is None:
        confusion_index = CONFUSION_INDEX
    if test_methods is None:
        test_methods = TEST_METHODS

    original_functions = {
        net_name: model.networks[net_name].function
        for net_name in model.networks
    }
    if test_functions:
        for net_name, new_func in test_functions.items():
            model.networks[net_name].function = new_func

    with torch.no_grad():
        confusion_dict = get_confusion_matrix(
            model, confusion_index, test_queries, test_methods
        )

    res_list = []
    for k, confusion in confusion_dict.items():
        f1 = calculate_f1(confusion)

        accuracy = calculate_accuracy(confusion)

        print(k)
        print(confusion)

        f1_text = 'F1 {}: {}'.format(k, f1)
        print(f1_text)

        res_list.append(
            ('F1 {}'.format(k), f1)
        )

        acc_text = 'Accuracy {}: {}'.format(k, accuracy)
        print(acc_text)

        res_list.append(
            ('Accuracy {}'.format(k), accuracy)
        )

    # Restore the original function
    for net_name, ori_func in original_functions.items():
        model.networks[net_name].function = ori_func

    return res_list
