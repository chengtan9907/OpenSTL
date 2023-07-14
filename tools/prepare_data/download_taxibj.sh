#!/usr/bin/env bash

# run this script on the root
mkdir data/taxibj
cd data/taxibj

# download dataset.npz in `data/taxibj/`
wget https://github.com/chengtan9907/OpenSTL/releases/download/v0.1.0/taxibj_dataset.zip
unzip taxibj_dataset.zip
rm taxibj_dataset.zip

echo "finished"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── taxibj
#     │   ├── dataset.npz
