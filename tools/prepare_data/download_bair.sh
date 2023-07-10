#!/usr/bin/env bash

# run this script on the root
cd data

# Download BAIR in tensorflow format
wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
tar -xvf bair_robot_pushing_dataset_v0.tar
rm bair_robot_pushing_dataset_v0.tar

echo "finished"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── softmotion30_44k
#     │   ├── test
#     │   ├── train
