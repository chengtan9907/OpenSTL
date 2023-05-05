#!/usr/bin/env bash

# run this script on the root
mkdir data/moving_mnist
cd data/moving_mnist

# download mmnist and place them in `data/moving_mnist/`
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

echo "finished"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── moving_mnist
#     │   ├── mnist_test_seq.npy
#     │   ├── train-images-idx3-ubyte.gz
