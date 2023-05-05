#!/usr/bin/env bash

# run this script on the root
mkdir data/moving_fmnist
cd data/moving_fmnist

# download fmnist and place them in `data/moving_fmnist/`
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz

# download the test data and untar it (or download from Baidu Cloud `https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk`)
wget https://github.com/chengtan9907/OpenSTL/releases/download/v0.1.0/fmnist_test_seq.npy.tar
tar -xf fmnist_test_seq.npy.tar

# # you can also generate the test data by yourself (not recommend)
# cd ../..
# python tools/prepare_data/generate_mmnist.py fmnist

echo "finished"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── moving_fmnist
#     │   ├── fmnist_test_seq.npy
#     │   ├── train-images-idx3-ubyte.gz
