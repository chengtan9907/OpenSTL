# Installation

## Install the project

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/chengtan9907/SimVPv2
cd SimVPv2
conda env create -f environment.yml
conda activate SimVP
python setup.py develop
```

<details close>
<summary>Dependencies</summary>

* argparse
* numpy
* hickle
* scikit-image=0.16.2
* torch
* timm
* tqdm
</details>

## Prepare datasets

It is recommended to symlink your dataset root (assuming `$YOUR_DATA_ROOT`) to `$OPENMIXUP/data`. If your folder structure is different, you need to change the corresponding paths in config files.

We support following datasets: [KTH Action](https://ieeexplore.ieee.org/document/1334462) [[download](https://www.csc.kth.se/cvap/actions/)], [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) [[download](https://figshare.com/articles/dataset/KITTI_hkl_files/7985684)], [Moving MNIST](http://arxiv.org/abs/1502.04681) [[download](http://www.cs.toronto.edu/~nitish/unsupervised_video/)], [TaxiBJ](https://arxiv.org/abs/1610.00081) [[download](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)], [WeatherBench](https://arxiv.org/abs/2002.00469) [[download](https://github.com/pangeo-data/WeatherBench)]. You can also download the version we used in experiments from [**Baidu Cloud**](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk) (kjfk). Please do not distribute the datasets or only use them for research.

```
SimVPv2
├── configs
└── data
    ├── caltech
    ├── human
    |   ├── dataset.npz
    |── kitti_hkl
    |   ├── sources_test_mini.hkl
    |   ├── X_train.hkl
    |   ├── ...
    |── kth
    |   ├── boxing
    |   ├── ...
    |── moving_fmnist
    |── moving_mnist
    |   ├── mnist_test_seq.npy
    |   ├── train-images-idx3-ubyte.gz
    |── taxibj
    |   ├── dataset.npz
    |── weather
    |   ├── 2m_temperature
    |   ├── ...
```
