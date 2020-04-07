from train import train_model
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
import torch
from torch import nn


FEATURES = {
    0: [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    1: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    2: [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    3: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    4: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    5: [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    6: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    7: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    8: [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    9: [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
}
# FEATURES = {
#     0: [0.7186, 0.1632, 0.3268, 0.2789, 0.8353, 0.9288],
#     1: [0.8955, 0.6153, 0.2256, 0.2843, 0.0807, 0.0664],
#     2: [0.7965, 0.6725, 0.6579, 0.2019, 0.1342, 0.8956],
#     3: [0.8018, 0.8409, 0.2927, 0.7286, 0.7260, 0.6589],
#     4: [0.4523, 0.5443, 0.9024, 0.1042, 0.7745, 0.2642],
#     5: [0.6084, 0.1411, 0.1454, 0.2567, 0.4549, 0.6486],
#     6: [0.0404, 0.8114, 0.4359, 0.8645, 0.4648, 0.7401],
#     7: [0.6837, 0.5738, 0.4132, 0.2002, 0.1748, 0.7729],
#     8: [0.3648, 0.2533, 0.1007, 0.5824, 0.6238, 0.7889],
#     9: [0.9991, 0.9765, 0.2481, 0.0740, 0.6851, 0.7006]
# }


CLASSES = {
    0: [1, 1, 0],
    1: [1, 0, 1],
    2: [0, 1, 1],
    3: [1, 0, 0],
    4: [1, 0, 1],
    5: [0, 1, 1],
    6: [1, 0, 1],
    7: [1, 0, 1],
    8: [0, 0, 1],
    9: [0, 0, 1]
}


CLASSES_BY_LABEL = {
    0: {'a': 1, 'b': 1, 'c': 0},
    1: {'a': 1, 'b': 0, 'c': 1},
    2: {'a': 0, 'b': 1, 'c': 1},
    3: {'a': 1, 'b': 0, 'c': 0},
    4: {'a': 1, 'b': 0, 'c': 1},
    5: {'a': 0, 'b': 1, 'c': 1},
    6: {'a': 1, 'b': 0, 'c': 1},
    7: {'a': 1, 'b': 0, 'c': 1},
    8: {'a': 0, 'b': 0, 'c': 1},
    9: {'a': 0, 'b': 0, 'c': 1}
}


class MultiLabelNet(nn.Module):
    def __init__(self):
        super(MultiLabelNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(6, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


def neural_predicate(network, d):
    i = torch.Tensor(FEATURES[d])

    output = network.net(i)

    return output.squeeze(0)


def multilabel_test(model):
    total_acc = 0

    for i, f in FEATURES.items():
        out = model.networks['multilabel_net'].net.forward(torch.Tensor(f))

        rounded = out.round()

        curr_acc = sum([int(t == p) for t, p in zip(CLASSES[i], rounded)]) / len(rounded)
        print(i, curr_acc)

        total_acc += curr_acc

    print("The total accuracy is {}".format(total_acc / len(FEATURES)))
    print("================================")

    return []


def run(training_data, test_data, problog_file):
    queries = load(training_data)
    test_queries = load(test_data)

    network = MultiLabelNet()

    with open(problog_file, 'r') as f:
        problog_string = f.read()

    net = Network(network, 'multilabel_net', neural_predicate)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    model = Model(problog_string, [net], caching=False)
    optimizer = Optimizer(model, 2)

    train_model(
        model,
        queries,
        nr_epochs=50,
        optimizer=optimizer,
        test_iter=len(queries) * 10,
        # test=multilabel_test,
        log_iter=500,
        snapshot_iter=len(queries)
    )

    for query in test_queries:
        print(query)

        for k, v in model.solve(query).items():
            print('\t{}: {:.4f}\t{}'.format(k.args[1], v[0], CLASSES_BY_LABEL[int(query.args[0])][str(k.args[1])]))


if __name__ == '__main__':
    run(
        'train_data_tiny.txt',
        'test_data_tiny.txt',
        'multilabel.pl'
    )
