# Installation

## Install the project

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/chengtan9907/OpenSTL
cd OpenSTL
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop  # or `pip install -e .`
```

<details close>
<summary>Requirements</summary>

* Linux (Windows is not officially supported)
* Python 3.7+
* PyTorch 1.8 or higher
* CUDA 10.1 or higher
* NCCL 2
* GCC 4.9 or higher
</details>

<details close>
<summary>Dependencies</summary>

* argparse
* dask
* decord
* fvcore
* hickle
* lpips
* matplotlib
* netcdf4
* numpy
* opencv-python
* packaging
* pandas
* python<=3.10.8
* scikit-image
* scikit-learn
* torch
* timm
* tqdm
* xarray==0.19.0
</details>

**Note:**

1. Installation errors. 
    * If you are installing `cv2` for the first time, `ImportError: libGL.so.1` will occur, which can be solved by `apt install libgl1-mesa-glx`.
    * Errors might occur with `hickle` and this dependency when using KittiCaltech dataset. You can solve the issues by installing additional packages according to the output message.
    * As for WeatherBench, you encounter some import or runtime errors in the version of `xarray`. You can install the latest version or `xarray==0.19.0` to solve the errors, i.e., `pip install xarray==0.19.0`, and then install required packages according to error messages.
    * Please use Python<=3.10.x to prevent the error of timm, `ValueError: mutable default <class 'timm.models.maxxvit.MaxxVitConvCfg'> for field conv_cfg is not allowed: use default_factory`. Refer to issue [#1530](https://github.com/huggingface/pytorch-image-models/issues/1530) in issue [#62](https://github.com/chengtan9907/OpenSTL/issues/62).

2. Following the above instructions, OpenSTL is installed on `dev` mode, any local modifications made to the code will take effect. You can install it by `pip install .` to use it as a PyPi package, and you should reinstall it to make the local modifications effect.

## Prepare datasets

It is recommended to symlink your dataset root (assuming `$YOUR_DATA_ROOT`) to `$OPENSTL/data`. If your folder structure is different, you need to change the corresponding paths in config files.

We support following datasets: [Human3.6M](http://vision.imar.ro/human3.6m/pami-h36m.pdf) [[download](http://vision.imar.ro/human3.6m/description.php)], [KTH Action](https://ieeexplore.ieee.org/document/1334462) [[download](https://www.csc.kth.se/cvap/actions/)], [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) [[download](https://figshare.com/articles/dataset/KITTI_hkl_files/7985684)], [Moving MNIST](http://arxiv.org/abs/1502.04681) [[download](http://www.cs.toronto.edu/~nitish/unsupervised_video/)], [TaxiBJ](https://arxiv.org/abs/1610.00081) [[download](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)], [WeatherBench](https://arxiv.org/abs/2002.00469) [[download](https://github.com/pangeo-data/WeatherBench)]. Please prepare datasets with tools and scripts under `tools/prepare_data`. You can also download the version we used in experiments from [**Baidu Cloud**](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk) (kjfk). Please do not distribute the datasets and only use them for research. After all, the related datasets under `$OPENSTL/data` will look like this:

```
OpenSTL
├── configs
└── data
    ├── caltech
    │   ├── set06
    │   ├── ...
    │   ├── set10
    │   ├── data_cache.npy
    │   ├── indices_cache.npy
    ├── human
    |   ├── images
    |   ├── test.txt
    |   ├── train.txt
    ├── kinetics400
    │   ├── annotations
    │   ├── replacement
    │   ├── test
    │   ├── train
    │   ├── val
    |── kitti_hkl
    |   ├── sources_test_mini.hkl
    |   ├── ...
    |   ├── X_train.hkl
    │   ├── X_val.hkl
    |── kth
    |   ├── boxing
    |   ├── ...
    |   ├── walking
    |── moving_fmnist
    |   ├── fmnist_test_seq.npy
    |   ├── train-images-idx3-ubyte.gz
    |── moving_mnist
    |   ├── mnist_test_seq.npy
    |   ├── train-images-idx3-ubyte.gz
    ├── softmotion30_44k
    │   ├── test
    │   ├── train
    |── taxibj
    |   ├── dataset.npz
    |── weather
    |   ├── 2m_temperature
    |   ├── ...
    |── weather_1_40625deg
    |   ├── 2m_temperature
    |   ├── ...
```

### Moving MNIST / FMNIST / MNIST-CIFAR / MNIST Noise

[Moving MNIST](http://arxiv.org/abs/1502.04681) and [Moving FMNIST](http://arxiv.org/abs/1502.04681) are toy datasets, which generate gray-scale videos (64x64 resolutions) with two objects. We provide [download_mmnist.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_mmnist.sh) and [download_mfmnist.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_mfmnist.sh), which download datasets from [MMNIST download](http://www.cs.toronto.edu/~nitish/unsupervised_video/) and [MFMNIST download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz). Note that the train set is generated online while the test set is fixed to ensure the consistency of evaluation results.
We provided the combised version of MNIST and CIFAR-10 and the noise versions of Moving MNIST (dynamic / missing / perceptual) in the dataset implementation.

### BAIR Robot Pushing

The BAIR dataset uses [BAIR Robot Pushing](https://arxiv.org/abs/1710.05268) as the train set (648960 videos) and the test set (3840 videos). We provide [download_bair.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_bair.sh) to prepare the datasets, and you can also download the data from [BAIR download](http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar). The data preprocessing of RGB videos (64x64 resolutions) and experiment settings are adopted from [PredRNN](https://github.com/thuml/predrnn-pytorch).

### KittiCaltech Pedestrian

The KittiCaltech Pedestrian dataset uses [Kitti Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) as the train set (2042 videos) and uses [Caltech Pedestrian](https://data.caltech.edu/records/f6rph-90m20) as the test set (1983 videos). We provide [download_kitticaltech.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_kitticaltech.sh) to prepare the datasets. The data preprocessing of RGB videos (128x160 resolutions) and experiment settings are adopted from [PredNet](https://github.com/coxlab/prednet).

### KTH Action

The [KTH Action](https://ieeexplore.ieee.org/document/1334462) dataset contains grey-scale videos (resizing 160x120 to 128x128 resolutions) of six types of human actions performed several times by 25 subjects in four different scenarios. It has 5200 and 3167 videos for the train and test sets and can be downloaded from [KTH download](https://www.csc.kth.se/cvap/actions/), which are in the `avi` format. For convinience, we use the image version released in [PredRNN](https://github.com/thuml/predrnn-pytorch) and provide [download_kth.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_kth.sh) to prepare the dataset. The data preprocessing and experiment settings are adopted from [KTH](https://ieeexplore.ieee.org/document/1334462) and [PredRNN](https://github.com/thuml/predrnn-pytorch).

### Human 3.6M

The [Human3.6M](http://vision.imar.ro/human3.6m/pami-h36m.pdf) dataset contains high-resolution videos (1024x1024 resolutions) of seventeen scenarios of human actions performed by eleven professional actors, which can be downloaded from [Human3.6M download](http://vision.imar.ro/human3.6m/description.php). We provide [download_human3.6m.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_human3.6m.sh) to prepare the dataset. We borrow the train and test splitting files from [STRPM](https://github.com/ZhengChang467/STRPM) but use 256x256 resolutions in our experiments.

### Kinetics-400

The [Kinetics-400](https://arxiv.org/abs/1705.06950) dataset contains real-world human action videos (around 256x320 resolutions) of 400 human actions classes, with at least 400 video clips for each action. Each clip lasts around 10s and is taken from a different YouTube video. It has 246534 and 39805 videos for the train and test sets, which can be downloaded from [Kinetics download](https://www.deepmind.com/open-source/kinetics). We provide [download_kinetics.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_kinetics.sh) to prepare the dataset according to [kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset). Similar to Human 3.6M, we use 256x256 resolutions in our experiments for faster training.

### WeatherBench

[WeatherBench](https://arxiv.org/abs/2002.00469) is the publicly available dataset for global weather prediction, which can be downloaded and processed from [WeatherBench download](https://github.com/pangeo-data/WeatherBench). We choose some important weather variants with certain vertical levels and resolutions, e.g., 2m_temperature, relative_humidity, and total_cloud_cover. You can download the specific dataset of WeatherBench with [download_weatherbench.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_weatherbench.sh). Note that `5.625deg` and `1.40625deg` indicate 32x64 and 128x256 resolutions, and the data can have multiple channels.

### TaxiBJ

[TaxiBJ](https://arxiv.org/abs/1610.00081) is a popular traffic trajectory prediction dataset, which contains the trajectory data (32x32) in Beijing collected from taxicab GPS with two channels, which can be downloaded from [OneDrive](https://1drv.ms/f/s!Akh6N7xv3uVmhOhDKwx3bm5zpHkDOQ). We provide [download_taxibj.sh](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data/download_taxibj.sh) to prepare the dataset, or you can download it from [Baidu Cloud](http://pan.baidu.com/s/1qYq7ja8). We borrow the data preprocessing scripts from [DeepST](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ) and provide the processed data in our [Baidu Cloud](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk).
