from train import batch_train_model
from data_loader import load
from examples.NIPS.MNIST.mnist import test_MNIST, MNIST_Net, neural_predicate
from model import Model
from optimizer import Optimizer
from network import Network
import torch


# queries = load('train_data.txt')
queries = load('aux_train_data.txt')

with open('addition.pl') as f:
    problog_string = f.read()


network = MNIST_Net()
net = Network(network, 'mnist_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(),lr = 0.01)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)

batch_train_model(model, queries, 10, optimizer, test=test_MNIST, batch_size=10)
