# Getting Started

This page provides basic tutorials about the usage of SimVP. For installation instructions, please see [Install](docs/en/install.md).

An example of single GPU training SimVP+gSTA on Moving MNIST dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python tools/non_dist_train.py -d mmnist -m SimVP --model_type gsta --lr 1e-3 --ex_name mmnist_simvp_gsta
```
