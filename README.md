# Introduction

This repository contains the code to perform Complex Event Processing (CEP) using an event calculus approach in DeepProbLog.

The implementation for the CEP with an event calculus approach is inspired by the paper [A Probabilistic Logic Programming Event Calculus](https://arxiv.org/abs/1204.1851).

The code for DeepProbLog has slightly modified from the original source, presented in the paper [DeepProbLog:
Neural Probabilistic Logic Programming](https://arxiv.org/abs/1805.10872) in the following ways:

 * Slight changes on how the training of the neural network is performed.
 * Code has been added to perform CEP on:
   * [MNIST digits](http://yann.lecun.com/exdb/mnist/)
   * [Urban Sounds 8K](https://urbansounddataset.weebly.com/urbansound8k.html)

# DeepProbLog

DeepProbLog is an extension of [ProbLog](https://dtai.cs.kuleuven.be/problog/) that integrates Probabilistic Logic Programming with Deep Learning. See our [paper](https://arxiv.org/abs/1805.10872) on DeepProbLog.

## Requirements

DeepProbLog has the following requirements:

* [ProbLog](https://dtai.cs.kuleuven.be/problog/)
* [PyTorch](https://pytorch.org/)

## Examples

DeepProbLog comes with a few examples:

* [MNIST addition](examples/NIPS/MNIST/)
* [Program Induction](examples/NIPS/Forth/)
* [The coin-urn experiment](examples/NIPS/CoinUrn/)
