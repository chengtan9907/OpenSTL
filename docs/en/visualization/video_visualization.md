# Video Prediction Visualization

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on popular traffic prediction datasets. More STL methods will be supported in the future. Issues and PRs are welcome!** Visualization of *GIF* is released.

## Table of Contents

- [Visualization of Moving MNIST Benchmarks](#visualization-of-moving-mnist-benchmarks)
- [Visualization of Moving FashionMNIST Benchmarks](#visualization-of-moving-fmnist-benchmarks)
- [Visualization of Moving MNIST-CIFAR Benchmarks](#visualization-of-moving-mnist-cifar-benchmarks)
- [Visualization of KittiCaltech Benchmarks](#visualization-of-kitticaltech-benchmarks)
- [Visualization of KTH Benchmarks](#visualization-of-kth-benchmarks)
- [Visualization of Human 3.6M Benchmarks](#visualization-of-human-36m-benchmarks)

<details open>
<summary>Currently supported spatiotemporal prediction methods</summary>

- [x] [ConvLSTM](https://arxiv.org/abs/1506.04214) (NeurIPS'2015)
- [x] [PredNet](https://openreview.net/forum?id=B1ewdt9xe) (ICLR'2017)
- [x] [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) (NeurIPS'2017)
- [x] [PredRNN++](https://arxiv.org/abs/1804.06300) (ICML'2018)
- [x] [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2018)
- [x] [MIM](https://arxiv.org/abs/1811.07490) (CVPR'2019)
- [x] [CrevNet](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2020)
- [x] [PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
- [x] [MAU](https://openreview.net/forum?id=qwtfY-3ibt7) (NeurIPS'2021)
- [x] [PredRNN.V2](https://arxiv.org/abs/2103.09504v4) (TPAMI'2022)
- [x] [SimVP](https://arxiv.org/abs/2206.05099) (CVPR'2022)
- [x] [SimVP.V2](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
- [x] [TAU](https://arxiv.org/abs/2206.12126) (CVPR'2023)
- [x] [DMVFN](https://arxiv.org/abs/2303.09875) (CVPR'2023)

</details>

<details open>
<summary>Currently supported MetaFormer models for SimVP</summary>

- [x] [ViT](https://arxiv.org/abs/2010.11929) (ICLR'2021)
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

We provide visualization figures of various video prediction methods on various benchmarks. You can plot your own visualization with tested results (e.g., `work_dirs/exp_name/saved`) by [vis_video.py](https://github.com/chengtan9907/OpenSTL/tree/master/tools/visualizations/vis_video.py). Note that `--vis_dirs` denotes visualize all experimental folders under the path, and `--vis_channel` can select the channel for visualization. For example, run plotting with the script:
```shell
python tools/visualizations/vis_video.py -d mmnist -w work_dirs/exp_name --index 0 --save_dirs fig_mmnist_vis
```

## Visualization of Moving MNIST Benchmarks

We provide benchmark results on the popular [Moving MNIST](http://arxiv.org/abs/1502.04681) dataset using $10\rightarrow 10$ frames prediction setting in [configs/mmnist](https://github.com/chengtan9907/OpenSTL/configs/mmnist).

| ConvLSTM | DMVFN |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_DMVFN.gif' height="auto" width="260" ></div> |

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>


## Visualization of Moving FMNIST Benchmarks

Similar to [Moving MNIST](http://arxiv.org/abs/1502.04681), we also provide the advanced version of MNIST, i.e., MFMNIST benchmark results, using $10\rightarrow 10$ frames prediction setting in [configs/mfmnist](https://github.com/chengtan9907/OpenSTL/configs/mfmnist).

| ConvLSTM |
| :---: | 
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_ConvLSTM.gif' height="auto" width="260" ></div> | 

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>

## Visualization of Moving MNIST-CIFAR Benchmarks

Similar to [Moving MNIST](http://arxiv.org/abs/1502.04681), we further design the advanced version of MNIST with complex backgrounds from CIFAR-10, i.e., MMNIST-CIFAR benchmark, using $10\rightarrow 10$ frames prediction setting in [configs/mmnist_cifar](https://github.com/chengtan9907/OpenSTL/configs/mmnist_cifar).

| ConvLSTM | 
| :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_ConvLSTM.gif' height="auto" width="260" ></div> | 

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>


## Visualization of KittiCaltech Benchmarks

We provide benchmark results on [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) dataset using $10\rightarrow 1$ frames prediction setting in [configs/kitticaltech](https://github.com/chengtan9907/OpenSTL/configs/kitticaltech).

| ConvLSTM | DMVFN |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_DMVFN.gif' height="auto" width="260" ></div> |

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>


## Visualization of KTH Benchmarks

We provide long-term prediction benchmark results on [KTH Action](https://ieeexplore.ieee.org/document/1334462) dataset using $10\rightarrow 20$ frames prediction setting in [configs/kth](https://github.com/chengtan9907/OpenSTL/configs/kth).

| ConvLSTM | DMVFN |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_DMVFN.gif' height="auto" width="260" ></div> |

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>


## Visualization of Human 3.6M Benchmarks

We further provide high-resolution benchmark results on [Human3.6M](http://vision.imar.ro/human3.6m/pami-h36m.pdf) dataset using $4\rightarrow 4$ frames prediction setting in [configs/human](https://github.com/chengtan9907/OpenSTL/configs/human).

| ConvLSTM | DMVFN |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_DMVFN.gif' height="auto" width="260" ></div> |

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>
