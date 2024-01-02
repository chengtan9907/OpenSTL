<p align="center" width="100%">
  <img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/246222783-fdda535f-e132-4fdd-8871-2408cd29a264.png' width="50%">
</p>

# OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning

<p align="left">
<a href="https://arxiv.org/abs/2306.11249" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2306.11249-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/blob/master/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<!-- <a href="https://huggingface.co/OpenSTL" alt="Huggingface">
    <img src="https://img.shields.io/badge/huggingface-OpenSTL-blueviolet" /></a> -->
<a href="https://openstl.readthedocs.io/en/latest/" alt="docs">
    <img src="https://readthedocs.org/projects/openstl/badge/?version=latest" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/chengtan9907/SimVPv2?color=%23FF9600" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="resolution">
    <img src="https://img.shields.io/badge/issue%20resolution-1%20d-%23B7A800" /></a>
<a href="https://img.shields.io/github/stars/chengtan9907/OpenSTL" alt="arXiv">
    <img src="https://img.shields.io/github/stars/chengtan9907/OpenSTL" /></a>
</p>

[📘Documentation](https://openstl.readthedocs.io/en/latest/) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](docs/en/model_zoos/video_benchmarks.md) |
[🤗Huggingface](https://huggingface.co/OpenSTL) |
[👀Visualization](docs/en/visualization/video_visualization.md) |
[🆕News](docs/en/changelog.md)

## Introduction

OpenSTL is a comprehensive benchmark for spatio-temporal predictive learning, encompassing a broad spectrum of methods and diverse tasks, ranging from synthetic moving object trajectories to real-world scenarios such as human motion, driving scenes, traffic flow, and weather forecasting. OpenSTL offers a modular and extensible framework, excelling in user-friendliness, organization, and comprehensiveness. The codebase is organized into three abstracted layers, namely the core layer, algorithm layer, and user interface layer, arranged from the bottom to the top. We support PyTorch Lightning implementation [OpenSTL-Lightning](https://github.com/chengtan9907/OpenSTL/tree/OpenSTL-Lightning) (recommended) and naive PyTorch version [OpenSTL](https://github.com/chengtan9907/OpenSTL/tree/OpenSTL).

<p align="center" width="100%">
  <img src='https://github.com/chengtan9907/OpenSTL/assets/34480960/4f466441-a78a-405c-beb6-00a37e3d3827' width="90%">
</p>
<!-- <p align="center" width="100%">
  <img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/246222226-61e6b8e8-959c-4bb3-a1cd-c994b423de3f.png' width="90%">
</p> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview

<details open>
<summary>Major Features and Plans</summary>

- **Flexiable Code Design.**
  OpenSTL decomposes STL algorithms into `methods` (training and prediction), `models` (network architectures), and `modules`, while providing unified experiment API. Users can develop their own STL algorithms with flexible training strategies and networks for different STL tasks.

- **Standard Benchmarks.**
  OpenSTL will support standard benchmarks of STL algorithms image with training and evaluation as many open-source projects (e.g., [MMDetection](https://github.com/open-mmlab/mmdetection) and [USB](https://github.com/microsoft/Semi-supervised-learning)). We are working on training benchmarks and will update results synchronizingly.

- **Plans.**
  We plan to provide benchmarks of various STL methods and MetaFormer architectures based on SimVP in various STL application tasks, e.g., video prediction, weather prediction, traffic prediction, etc. We encourage researchers interested in STL to contribute to OpenSTL or provide valuable advice!

</details>

<details open>
<summary>Code Structures</summary>

- `openstl/api` contains an experiment runner.
- `openstl/core` contains core training plugins and metrics.
- `openstl/datasets` contains datasets and dataloaders.
- `openstl/methods/` contains training methods for various video prediction methods.
- `openstl/models/` contains the main network architectures of various video prediction methods.
- `openstl/modules/` contains network modules and layers.
- `tools/` contains the executable python files `tools/train.py` and `tools/test.py` with possible arguments for training, validating, and testing pipelines.

</details>

## News and Updates

[2023-12-15] [OpenSTL-Lightning](https://github.com/chengtan9907/OpenSTL/tree/OpenSTL-Lightning) (`OpenSTL` v1.0.0) is released.

[2023-09-23] The OpenSTL paper has been accepted by NeurIPS 2023 Dataset and Benchmark Track! [arXiv](https://arxiv.org/abs/2306.11249) / [Zhihu](https://zhuanlan.zhihu.com/p/640271275).

[2023-06-19] `OpenSTL` v0.3.0 is released and will be enhanced in [#25](https://github.com/chengtan9907/OpenSTL/issues/25).

## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/chengtan9907/OpenSTL
cd OpenSTL
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```

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

Please refer to [install.md](docs/en/install.md) for more detailed instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage. Here is an example of single GPU non-distributed training SimVP+gSTA on Moving MNIST dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python tools/train.py -d mmnist --lr 1e-3 -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

## Tutorial on using Custom Data

For the convenience of users, we provide a tutorial on how to train, evaluate, and visualize with OpenSTL on custom data. This tutorial enables users to quickly build their own projects using OpenSTL. For more details, please refer to the [`tutorial.ipynb`](examples/tutorial.ipynb) in the `examples/` directory.

We also provide a Colab demo of this tutorial:

<a href="https://colab.research.google.com/drive/19uShc-1uCcySrjrRP3peXf2RUNVzCjHh?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview of Model Zoo and Datasets

We support various spatiotemporal prediction methods and provide [benchmarks](https://github.com/chengtan9907/OpenSTL/tree/master/docs/en/model_zoos) on various STL datasets. We are working on add new methods and collecting experiment results.

* Spatiotemporal Prediction Methods.

    <details open>
    <summary>Currently supported methods</summary>

    - [x] [ConvLSTM](https://arxiv.org/abs/1506.04214) (NeurIPS'2015)
    - [x] [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) (NeurIPS'2017)
    - [x] [PredRNN++](https://arxiv.org/abs/1804.06300) (ICML'2018)
    - [x] [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2018)
    - [x] [MIM](https://arxiv.org/abs/1811.07490) (CVPR'2019)
    - [x] [PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
    - [x] [MAU](https://openreview.net/forum?id=qwtfY-3ibt7) (NeurIPS'2021)
    - [x] [PredRNN.V2](https://arxiv.org/abs/2103.09504v4) (TPAMI'2022)
    - [x] [SimVP](https://arxiv.org/abs/2206.05099) (CVPR'2022)
    - [x] [SimVP.V2](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
    - [x] [TAU](https://arxiv.org/abs/2206.12126) (CVPR'2023)
    - [x] [MMVP](https://arxiv.org/abs/2308.16154) (ICCV'2023)
    - [x] [SwinLSTM](https://arxiv.org/abs/2308.09891) (ICCV'2023)
    - [x] WaST (AAAI'2024)

    </details>

    <details open>
    <summary>Currently supported MetaFormer models for SimVP</summary>

    - [x] [ViT (Vision Transformer)](https://arxiv.org/abs/2010.11929) (ICLR'2021)
    - [x] [Swin-Transformer](https://arxiv.org/abs/2103.14030) (ICCV'2021)
    - [x] [MLP-Mixer](https://arxiv.org/abs/2105.01601) (NeurIPS'2021)
    - [x] [ConvMixer](https://arxiv.org/abs/2201.09792) (Openreview'2021)
    - [x] [UniFormer](https://arxiv.org/abs/2201.09450) (ICLR'2022)
    - [x] [PoolFormer](https://arxiv.org/abs/2111.11418) (CVPR'2022)
    - [x] [ConvNeXt](https://arxiv.org/abs/2201.03545) (CVPR'2022)
    - [x] [VAN](https://arxiv.org/abs/2202.09741) (ArXiv'2022)
    - [x] [IncepU (SimVP.V1)](https://arxiv.org/abs/2206.05099) (CVPR'2022)
    - [x] [gSTA (SimVP.V2)](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
    - [x] [HorNet](https://arxiv.org/abs/2207.14284) (NeurIPS'2022)
    - [x] [MogaNet](https://arxiv.org/abs/2211.03295) (ArXiv'2022)

    </details>

* Spatiotemporal Predictive Learning Benchmarks ([prepare_data](https://github.com/chengtan9907/OpenSTL/tree/master/tools/prepare_data) or [Baidu Cloud](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk)).

    <details open>
    <summary>Currently supported datasets</summary>

    - [x] [BAIR Robot Pushing](https://arxiv.org/abs/1710.05268) (CoRL'2017) [[download](https://sites.google.com/berkeley.edu/robotic-interaction-datasets)] [[config](configs/bair)]
    - [x] [Human3.6M](http://vision.imar.ro/human3.6m/pami-h36m.pdf) (TPAMI'2014) [[download](http://vision.imar.ro/human3.6m/description.php)] [[config](configs/human)]
    - [x] [KTH Action](https://ieeexplore.ieee.org/document/1334462) (ICPR'2004) [[download](https://www.csc.kth.se/cvap/actions/)] [[config](configs/kth)]
    - [x] [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) (IJRR'2013) [[download](https://www.dropbox.com/s/rpwlnn6j39jjme4/kitti_data.zip)] [[config](configs/kitticaltech)]
    - [x] [Kinetics-400](https://arxiv.org/abs/1705.06950) (ArXiv'2017) [[download](https://deepmind.com/research/open-source/kinetics)] [[config](configs/kinetics)]
    - [x] [Moving MNIST](http://arxiv.org/abs/1502.04681) (ICML'2015) [[download](http://www.cs.toronto.edu/~nitish/unsupervised_video/)] [[config](configs/mmnist)]
    - [x] [Moving FMNIST](http://arxiv.org/abs/1502.04681) (ICML'2015) [[download](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk)] [[config](configs/mfmnist)]
    - [x] [TaxiBJ](https://arxiv.org/abs/1610.00081) (AAAI'2017) [[download](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)] [[config](configs/taxibj)]
    - [x] [WeatherBench](https://arxiv.org/abs/2002.00469) (ArXiv'2020) [[download](https://github.com/pangeo-data/WeatherBench)] [[config](configs/weather)]

    </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## Visualization

We present visualization examples of ConvLSTM below. For more detailed information, please refer to the [visualization](docs/en/visualization/).

- For synthetic moving object trajectory prediction and real-world video prediction, visualization examples of other approaches can be found in [visualization/video_visualization.md](docs/en/visualization/video_visualization.md). BAIR and Kinetics are not benchmarked and only for illustration.

- For traffic flow prediction, visualization examples of other approaches are shown in [visualization/traffic_visualization.md](docs/en/visualization/traffic_visualization.md).

- For weather forecasting, visualization examples of other approaches are shown in [visualization/weather_visualization.md](docs/en/visualization/weather_visualization.md).

<div align="center">

| Moving MNIST | Moving FMNIST | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_ConvLSTM.gif' height="auto" width="260" ></div> |

| Moving MNIST-CIFAR | KittiCaltech |
| :---: | :---: |
|  <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_ConvLSTM.gif' height="auto" width="260" ></div> |

| KTH | Human 3.6M | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_ConvLSTM.gif' height="auto" width="260" ></div> |

| Traffic - in flow | Traffic - out flow |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_in_flow_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_out_flow_ConvLSTM.gif' height="auto" width="260" ></div> |

| Weather - Temperature | Weather - Humidity |
|  :---: |  :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_ConvLSTM.gif' height="auto" width="360" ></div>|

| Weather - Latitude Wind | Weather - Cloud Cover | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_ConvLSTM.gif' height="auto" width="360" ></div> |

| BAIR Robot Pushing | Kinetics-400 | 
| :---: | :---: |
| <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872182-4f31928d-2ebc-4407-b2d4-1fe4a8da5837.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872560-00775edf-5773-478c-8836-f7aec461e209.gif' height="auto" width="260" ></div> |

</div>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

## Acknowledgement

OpenSTL is an open-source project for STL algorithms created by researchers in **CAIRI AI Lab**. We encourage researchers interested in video and weather prediction to contribute to OpenSTL! We borrow the official implementations of [ConvLSTM](https://arxiv.org/abs/1506.04214), [PredNet](https://arxiv.org/abs/1605.08104), [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) variants, [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX), [MAU](https://arxiv.org/abs/1811.07490), [PhyDNet](https://arxiv.org/abs/2003.01460), [MMVP](https://arxiv.org/abs/2308.16154), and [SwinLSTM](https://arxiv.org/abs/2308.09891).

## Citation

If you are interested in our repository or our paper, please cite the following paper:

```
@inproceedings{tan2023openstl,
  title={OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning},
  author={Tan, Cheng and Li, Siyuan and Gao, Zhangyang and Guan, Wenfei and Wang, Zedong and Liu, Zicheng and Wu, Lirong and Li, Stan Z},
  booktitle={Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}

@inproceedings{gao2022simvp,
  title={Simvp: Simpler yet better video prediction},
  author={Gao, Zhangyang and Tan, Cheng and Wu, Lirong and Li, Stan Z},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3170--3180},
  year={2022}
}
@article{tan2022simvpv2,
  title={SimVP: Towards Simple yet Powerful Spatiotemporal Predictive Learning},
  author={Tan, Cheng and Gao, Zhangyang and Li, Siyuan and Li, Stan Z},
  journal={arXiv preprint arXiv:2211.12509},
  year={2022}
}
@inproceedings{tan2023temporal,
  title={Temporal attention unit: Towards efficient spatiotemporal predictive learning},
  author={Tan, Cheng and Gao, Zhangyang and Wu, Lirong and Xu, Yongjie and Xia, Jun and Li, Siyuan and Li, Stan Z},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18770--18782},
  year={2023}
}
```

## Contribution and Contact

For adding new features, looking for helps, or reporting bugs associated with `OpenSTL`, please open a [GitHub issue](https://github.com/chengtan9907/OpenSTL/issues) and [pull request](https://github.com/chengtan9907/OpenSTL/pulls) with the tag "new features", "help wanted", or "enhancement". Feel free to contact us through email if you have any questions.

- Siyuan Li (lisiyuan@westlake.edu.cn), Westlake University & Zhejiang University
- Cheng Tan (tancheng@westlake.edu.cn), Westlake University & Zhejiang University
- Zhangyang Gao (gaozhangyang@westlake.edu.cn), Westlake University & Zhejiang University

<p align="right">(<a href="#top">back to top</a>)</p>
