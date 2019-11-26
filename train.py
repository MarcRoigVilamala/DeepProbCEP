import torch
import math
import signal
from logger import Logger
import time
from logic import term2list2
import random
import numpy as np

interrupt = False
zero_probability = False


def signal_handler(sig, frame):
        global interrupt
        print("Interrupted!")
        interrupt = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def handle_query(model, query, eps=1e-8):
    pos = True

    if type(query) is tuple:
        pos = query[1]
        query = query[0]
    ground = model.solve(query)
    for k in ground:
        if k == query:
            p, d = ground[k]
            break
    if pos:
        if p < 0:
            print('negative probability query:', query, p)
        # if p <= 0:
        #     print('zero probability query:', query, p)
        loss_grad = -1.0 / (p + eps)
        loss = -math.log(p + eps)
    else:
        loss_grad = 1.0 / (1.0 - p + eps)
        loss = -math.log(1 - p + eps)

    return loss_grad, loss, d


def train(model, optimizer, query, eps=1e-8, use_cuda=False, weights=None):
    loss_grad, loss, d = handle_query(model, query, eps)

    for k, v in d.items():
        if type(k[0]) is str:
            name = k[0]
            i = tuple(term2list2(k[1]))
            grad = loss_grad * v

            if weights is not None:
                grad = grad * weights

            grad = torch.FloatTensor(grad)
            if use_cuda:
                grad = grad.cuda()
            optimizer.backward(name,i,grad)
        else:
            optimizer.add_param_grad(k,loss_grad*float(v))

    return loss


def train_batch(model, optimizer, batch_queries, eps=1e-8, use_cuda=False):
    loss = 0

    gradients = {}

    for query in batch_queries:
        loss_grad, q_loss, d = handle_query(model, query, eps)
        loss += q_loss

        for k, v in d.items():
            if type(k[0]) is str:
                name = k[0]
                i = tuple(term2list2(k[1]))

                gradients[(name, i)] = loss_grad * v
            else:
                optimizer.add_param_grad(k, loss_grad * float(v))

    for (name, i), grad in gradients.items():
        grad = torch.FloatTensor(grad)
        if use_cuda:
            grad = grad.cuda()
        optimizer.backward(name, i, grad)

    optimizer.step()

    return loss


def train_model(model,queries,nr_epochs,optimizer, loss_function = train, test_iter=1000,test=None,log_iter=100,snapshot_iter=None,snapshot_name='model',shuffle=True):
    signal.signal(signal.SIGINT, signal_handler)
    i = 1
    accumulated_loss = 0
    logger = Logger()
    start = time.time()
    print("Training for {} epochs ({} iterations).".format(nr_epochs,nr_epochs*len(queries)))
    if test is not None:
        logger.log_list(i,test(model))
    for epoch in range(nr_epochs):
        epoch_start = time.time()
        if interrupt:
            break
        print("Epoch",epoch+1)
        q_indices = list(range(len(queries)))
        if shuffle:
            random.shuffle(q_indices)
        for q in q_indices:
            q = queries[q]
            iter_time = time.time()
            if interrupt:
                break
            loss = loss_function(model, optimizer, q)
            accumulated_loss += loss
            optimizer.step()
            if snapshot_iter and i % snapshot_iter == 0:
                fname = '{}_iter_{:06d}.mdl'.format(snapshot_name,i)
                print('Writing snapshot to '+fname)
                model.save_state(fname)
            if i % log_iter == 0:
                print('Iteration: ',i,'\tAverage Loss: ',accumulated_loss/log_iter)
                logger.log('time',i,iter_time - start)
                logger.log('loss',i,accumulated_loss/log_iter)
                for k in model.parameters:
                    logger.log(str(k),i,model.parameters[k])
                accumulated_loss = 0
            if test is not None and i % test_iter == 0:
                logger.log_list(i,test(model))

                # Clear all the evaluated values during testing because they may interfere with the training. If
                # no_grad has been used during testing and the same query is used for training it may be a problem.
                optimizer.clear()
            i += 1
        optimizer.step_epoch()
        print('Epoch time: ',time.time()-epoch_start)
    return logger


def batch_train_model(model, queries, nr_epochs, optimizer, loss_function=train_batch, test=None,
                      snapshot_name=None, shuffle=True, batch_size=64):
    signal.signal(signal.SIGINT, signal_handler)
    logger = Logger()
    start = time.time()
    print(
        "Training for {} epochs ({} iterations) with batches of {}.".format(
            nr_epochs, nr_epochs * len(queries), batch_size
        )
    )

    if test is not None:
        logger.log_list(0, test(model))

    for epoch in range(1, nr_epochs + 1):
        epoch_start = time.time()

        if interrupt:
            break

        print("Epoch", epoch)

        n_queries = len(queries)
        q_indices = list(range(n_queries))
        if shuffle:
            random.shuffle(q_indices)

        for beg_i in range(0, n_queries, batch_size):
            if interrupt:
                break

            q_batch = [
                q
                for j, q in enumerate(queries)
                if j in q_indices[beg_i:beg_i + batch_size]
            ]

            loss = loss_function(model, optimizer, q_batch)

            print(
                'Epoch: ', epoch,
                '\tBatch Start: ', beg_i,
                '\tLoss: ', loss,
                '\tAverage Loss: ', loss / len(q_batch)
            )
            # logger.log('time', i, iter_time - start)
            # logger.log('loss', i, loss / log_iter)
            # for k in model.parameters:
            #     logger.log(str(k), i, model.parameters[k])

        if snapshot_name:
            fname = '{}_epoch_{:04d}.mdl'.format(snapshot_name, epoch)
            print('Writing snapshot to ' + fname)
            model.save_state(fname)

        if test is not None:
            logger.log_list(epoch, test(model))

            # Clear all the evaluated values during testing because they may interfere with the training. If
            # no_grad has been used during testing and the same query is used for training it may be a problem.
            optimizer.clear()

        optimizer.step_epoch()
        print('Epoch time: ', time.time()-epoch_start)
    return logger
