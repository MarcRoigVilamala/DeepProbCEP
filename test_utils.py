import numpy as np
from tqdm import tqdm


def get_confusion_matrix(model, labels_dict, test_queries, methods_dict):
    confusion_dict = {}

    for query in tqdm(test_queries):
        get_target, query_transformation, get_result = methods_dict[query.functor]

        target = get_target(query)

        query = query_transformation(query)

        output = model.solve(query, evidence=None, test=True)

        model.clear_networks()

        result = get_result(output)

        labels = labels_dict[query.functor]

        if query.functor not in confusion_dict:
            confusion_dict[query.functor] = np.zeros(
                (len(labels), len(labels)), dtype=np.uint32
            )  # First index actual, second index predicted

        confusion = confusion_dict[query.functor]

        if result in labels_dict[query.functor]:
            confusion[labels[target], labels[result]] += 1

    return confusion_dict


def calculate_f1(confusion):
    f1 = np.zeros(len(confusion))
    for nr in range(len(confusion)):
        tp = confusion[nr, nr]
        fp = sum(confusion[:, nr]) - tp
        fn = sum(confusion[nr, :]) - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1[nr] = 2 * precision * recall / (precision + recall)

    return f1


def calculate_accuracy(confusion):
    return float(np.trace(confusion)) / np.sum(confusion)
