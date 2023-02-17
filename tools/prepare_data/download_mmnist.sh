#!/usr/bin/env bash

# run this script on the root
mkdir data/moving_mnist
cd data/moving_mnist

# down mmnist and place them in `data/moving_mnist/`
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
