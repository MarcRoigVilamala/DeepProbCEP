from TwoOutput3DResNet.Transformations.spatial_transforms import get_spatial_transform, get_norm_method

import sys

from examples.NIPS.ActivityDetection.prob_ec_testing import test

sys.path.append('../../../')
from train import train_model, train
from data_loader import load
from examples.NIPS.ActivityDetection.ActivityDetection import neural_predicate
from TwoOutput3DResNet.opts import parse_opts
from TwoOutput3DResNet.model import generate_model
from model import Model
from optimizer import Optimizer
from network import Network
import torch


def run(training_data, test_data, problog_files):
    queries = load(training_data)
    test_queries = load(test_data)

    problog_string = ''
    for problog_file in problog_files:
        with open(problog_file) as f:
            problog_string += f.read()
            problog_string += '\n\n'

    opt = parse_opts()
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    network, parameters = generate_model(opt)

    print(network)

    norm_method = get_norm_method(opt)

    train_spatial_transform = get_spatial_transform(opt, norm_method, 'train')
    test_spatial_transform = get_spatial_transform(opt, norm_method, 'test')

    net = Network(
        network,
        'activity_detection_net',
        lambda *args, **kwargs: neural_predicate(*args, **kwargs, spatial_transform=train_spatial_transform)
        # neural_predicate
    )
    # net = Network(network, 'activity_detection_net', neural_predicate)

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    # net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    net.optimizer = torch.optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov
    )
    model = Model(problog_string, [net], caching=False)
    # model.load_state('snapshots/activity_detection_epoch_0008.mdl')
    optimizer = Optimizer(model, 2)

    # train_model(model, queries, 1, optimizer, test_iter=1000, test=None, snapshot_iter=10000)
    train_model(
        model,
        queries,
        opt.n_epochs,
        optimizer,
        test_iter=len(queries),
        log_iter=min(100, len(queries)),
        # test=None,
        test=lambda model_to_test: test(
            model_to_test,
            test_queries,
            test_functions={
                'activity_detection_net': lambda *args, **kwargs: neural_predicate(
                    *args, **kwargs, spatial_transform=test_spatial_transform
                )
            },

        ),
        snapshot_iter=1000,
        snapshot_name='snapshots/activity_detection',
        loss_function=lambda *args, **kwargs: train(*args, **kwargs, use_cuda=True),
        shuffle=True
    )
    # batch_train_model(
    #     model,
    #     queries,
    #     opt.n_epochs,
    #     optimizer,
    #     # test=None,
    #     test=lambda model_to_test: test(
    #         model_to_test,
    #         test_queries,
    #         {
    #             'activity_detection_net': lambda *args, **kwargs: neural_predicate(
    #                 *args, **kwargs, spatial_transform=test_spatial_transform
    #             )
    #         }
    #     ),
    #     snapshot_name='snapshots/activity_detection',
    #     loss_function=lambda *args, **kwargs: train_batch(*args, **kwargs, use_cuda=True),
    #     shuffle=True,
    #     batch_size=20
    # )
    # test(model, test_queries)


if __name__ == '__main__':
    run(
        'init_training03.txt',
        'init_testing02.txt',
        ['ProbLogFiles/prob_ec_cached.pl', 'ProbLogFiles/event_defs.pl']
    )
