# SimVP: Towards Simple yet Powerful Spatiotemporal Predictive learning

<p align="left">
<a href="https://arxiv.org/abs/2211.12509" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2211.12509-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/Westlake-AI/MogaNet/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<a href="https://simvpv2.readthedocs.io/en/latest/" alt="docs">
    <img src="https://readthedocs.org/projects/simvpv2/badge/?version=latest" /></a>
<a href="https://simvpv2.readthedocs.io/en/latest/" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/chengtan9907/SimVPv2?color=%23FF9600" /></a>
<a href="https://github.com/chengtan9907/SimVPv2/issues" alt="resolution">
    <img src="https://img.shields.io/badge/issue%20resolution-1%20d-%23B7A800" /></a>
</p>

[📘Documentation](https://simvpv2.readthedocs.io/en/latest/) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](docs/en/model_zoos/video_benchmarks.md) |
[🆕News](docs/en/changelog.md)

This repository is an open-source project for video prediction benchmarks, which contains the implementation code for paper:

**SimVP: Towards Simple yet Powerful Spatiotemporal Predictive learning**  
[Cheng Tan](https://chengtan9907.github.io/), [Zhangyang Gao](https://scholar.google.com/citations?user=4SclT-QAAAAJ&hl=en), [Siyuan Li](https://lupin1998.github.io/), [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl).

## Introduction

This is the journal version of our previous conference work ([SimVP: Simpler yet Better Video Prediction](https://arxiv.org/abs/2206.05099), In CVPR 2022).

It is worth noticing that the hidden Translator $h$ in SimVP can be replaced by any [MetaFormer](https://arxiv.org/abs/2111.11418) block (satisfying the macro design of `token mixing` and `channel mixing`).
<p align="center">
    <img width="75%" src="https://user-images.githubusercontent.com/44519745/219116763-bb4ab408-7f25-47d3-b93e-79834c7b065e.png"> <br>
</p>

The performance of SimVPs on the Moving MNIST dataset. For the training time, the less the better. For the inference efficiency (frames per second), the more the better.
<p align="center">
    <img width="70%" src="https://user-images.githubusercontent.com/44519745/219116933-a34f5133-b38f-4657-bea4-c49361696fd2.png"> <br>
</p>

Quantitative results of different methods on the Moving MNIST dataset ($10 \rightarrow 10$ frames).
<p align="center">
    <img width="75%" src="https://user-images.githubusercontent.com/44519745/219117248-90825af9-28e8-42c6-8fec-858ef7f6dd92.png"> <br>
</p>

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview

* `simvp/api` contains an experiment runner.
* `simvp/core` contains core training plugins and metrics.
* `simvp/datasets` contains datasets and dataloaders.
* `simvp/methods/` contains training methods for various video prediction methods.
* `simvp/models/` contains the main network architectures of various video prediction methods.
* `simvp/modules/` contains network modules and layers.
* `tools/non_dist_train.py` is the executable python file with possible arguments for training, validating, and testing pipelines.

## News and Updates

[2023-02-18] `SimVP` v0.1.0 is released. Benchmark results and config files are updated for MMNIST, TaxiBJ, and WeatherBench datasets.

## Installation

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
* fvcore
* numpy
* hickle
* scikit-image=0.16.2
* scikit-learn
* torch
* timm
* tqdm
</details>

Please refer to [install.md](docs/en/install.md) for more detailed instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage. Here is an example of single GPU training SimVP+gSTA on Moving MNIST dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python tools/non_dist_train.py -d mmnist --lr 1e-3 -c ./configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview of Model Zoo

We support various video prediction methods and will provide benchmarks on various video prediction datasets. We are working on add new methods and collecting experiment results.

* Video Prediction Methods.

    <details open>
    <summary>Currently supported methods</summary>

    - [x] [ConvLSTM](https://arxiv.org/abs/1506.04214) (NIPS'2015)
    - [x] [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) (NIPS'2017)
    - [x] [PredRNN++](https://arxiv.org/abs/1804.06300) (ICML'2018)
    - [x] [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2018)
    - [x] [MAU](https://arxiv.org/abs/1811.07490) (CVPR'2019)
    - [x] [CrevNet](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2020)
    - [x] [PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
    - [x] [PredRNN.V2](https://arxiv.org/abs/2103.09504v4) (TPAMI'2022)
    - [x] [SimVP](https://arxiv.org/abs/2206.05099) (CVPR'2022)
    - [x] [SimVP.V2](https://arxiv.org/abs/2211.12509) (ArXiv'2022)

    </details>

    <details open>
    <summary>Currently supported MetaFormer models for SimVP</summary>

    - [x] [ViT](https://arxiv.org/abs/2010.11929) (ICLR'2021)
    - [x] [Swin-Transformer](https://arxiv.org/abs/2103.14030) (ICCV'2021)
    - [x] [MLP-Mixer](https://arxiv.org/abs/2105.01601) (NIPS'2021)
    - [x] [ConvMixer](https://arxiv.org/abs/2201.09792) (Openreview'2021)
    - [x] [UniFormer](https://arxiv.org/abs/2201.09450) (ICLR'2022)
    - [x] [PoolFormer](https://arxiv.org/abs/2111.11418) (CVPR'2022)
    - [x] [ConvNeXt](https://arxiv.org/abs/2201.03545) (CVPR'2022)
    - [x] [VAN](https://arxiv.org/abs/2202.09741) (ArXiv'2022)
    - [x] [IncepU (SimVP.V1)](https://arxiv.org/abs/2206.05099) (CVPR'2022)
    - [x] [gSTA (SimVP.V2)](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
    - [x] [HorNet](https://arxiv.org/abs/2207.14284) (NIPS'2022)
    - [x] [MogaNet](https://arxiv.org/abs/2211.03295) (ArXiv'2022)

    </details>

* Video Prediction Benchmarks.

    <details open>
    <summary>Currently supported datasets</summary>

    - [x] [KTH Action](https://ieeexplore.ieee.org/document/1334462) (ICPR'2004)  [[download](https://www.csc.kth.se/cvap/actions/)] [[config](https://github.com/chengtan9907/SimVPv2/configs/kth)]
    - [x] [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) (IJRR'2013) [[download](https://figshare.com/articles/dataset/KITTI_hkl_files/7985684)] [[config](https://github.com/chengtan9907/SimVPv2/configs/kitticaltech)]
    - [x] [Moving MNIST](http://arxiv.org/abs/1502.04681) (ICML'2015) [[download](http://www.cs.toronto.edu/~nitish/unsupervised_video/)] [[config](https://github.com/chengtan9907/SimVPv2/configs/mmnist)]
    - [x] [TaxiBJ](https://arxiv.org/abs/1610.00081) (AAAI'2017) [[download](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)] [[config](https://github.com/chengtan9907/SimVPv2/configs/taxibj)]
    - [x] [WeatherBench](https://arxiv.org/abs/2002.00469) (ArXiv'2020) [[download](https://github.com/pangeo-data/WeatherBench)] [[config](https://github.com/chengtan9907/SimVPv2/configs/weather)]

    </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

## Acknowledgement

SimVPv2 is an open-source project for video prediction methods created by researchers in **CAIRI AI Lab**. We encourage researchers interested in video and weather prediction to contribute to SimVPv2! We borrow the official implementations of [ConvLSTM](https://arxiv.org/abs/1506.04214), [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) variants, [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX), [MAU](https://arxiv.org/abs/1811.07490), [CrevNet](https://openreview.net/forum?id=B1lKS2AqtX), and [PhyDNet](https://arxiv.org/abs/2003.01460).

## Citation

If you are interested in our repository and our paper, please cite the following paper:

```
@article{tan2022simvp,
  title={SimVP: Towards Simple yet Powerful Spatiotemporal Predictive Learning},
  author={Tan, Cheng and Gao, Zhangyang and Li, Stan Z},
  journal={arXiv preprint arXiv:2211.12509},
  year={2022}
}
```

## Contribution and Contact

For adding new features, looking for helps, or reporting bugs associated with SimVPv2, please open a [GitHub issue](https://github.com/chengtan9907/SimVPv2/issues) and [pull request](https://github.com/chengtan9907/SimVPv2/pulls) with the tag "help wanted" or "enhancement". Feel free to contact us through email if you have any questions. Enjoy!

- Cheng Tan (tancheng@westlake.edu.cn), Westlake University & Zhejiang University
- Siyuan Li (lisiyuan@westlake.edu.cn), Westlake University & Zhejiang University
- Zhangyang Gao (gaozhangyang@westlake.edu.cn), Westlake University & Zhejiang University

<p align="right">(<a href="#top">back to top</a>)</p>
