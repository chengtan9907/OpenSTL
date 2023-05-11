# Weather Prediction Benchmarks

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on the famous weather prediction datasets, WeatherBench. More STL methods will be supported in the future. Issues and PRs are welcome!**

<details open>
<summary>Currently supported spatiotemporal prediction methods</summary>

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


## WeatherBench Benchmarks

We provide temperature prediction benchmark results on the popular [WeatherBench](https://arxiv.org/abs/2002.00469) dataset (temperature prediction `t2m`) using $12\rightarrow 12$ frames prediction setting. Metrics (MSE, MAE, SSIM, pSNR) of the final models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Cosine Annealing scheduler (no warmup and min lr is 1e-6).

### **STL Benchmarks on Temperature (t2m)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU**. We provide config files in [configs/weather/simvp_t2m_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/simvp_t2m_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| MetaFormer       |  Setting | Params | FLOPs |  FPS |  MSE  |   MAE  |  RMSE |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:-----:|:------:|:-----:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 1.521 | 0.7949 | 1.233 | model \| log |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 1.331 | 0.7246 | 1.154 | model \| log |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 |       |        |       | model \| log |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 |       |        |       | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 1.238 | 0.7037 | 1.113 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 1.105 | 0.6567 | 1.051 | model \| log |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 1.146 | 0.6712 | 1.070 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 1.143 | 0.6735 | 1.069 | model \| log |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 1.204 | 0.6885 | 1.097 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 1.255 | 0.7011 | 1.119 | model \| log |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 1.267 | 0.7073 | 1.126 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 1.156 | 0.6715 | 1.075 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 1.277 | 0.7220 | 1.130 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 1.150 | 0.6803 | 1.072 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 1.201 | 0.6906 | 1.096 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 1.152 | 0.6665 | 1.073 | model \| log |

### **STL Benchmarks on Humidity (r)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU**. We provide config files in [configs/weather/simvp_r_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/simvp_r_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| MetaFormer       |  Setting | Params | FLOPs |  FPS |  MSE   |   MAE  |  RMSE |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:------:|:------:|:-----:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 35.146 |  4.012 | 5.928 | model \| log |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 37.611 |  4.096 | 6.133 | model \| log |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 | 35.146 |  4.012 | 5.928 | model \| log |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 | 36.508 |  4.087 | 6.042 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 34.355 |  3.994 | 5.861 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 31.426 |  3.765 | 5.606 | model \| log |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 32.616 |  3.852 | 5.711 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 31.332 |  3.776 | 5.597 | model \| log |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 32.199 |  3.864 | 5.674 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 34.467 |  3.950 | 5.871 | model \| log |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 32.829 |  3.909 | 5.730 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 31.989 |  3.803 | 5.656 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 33.179 |  3.928 | 5.760 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 31.712 |  3.812 | 5.631 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 32.081 |  3.826 | 5.664 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 31.795 |  3.816 | 5.639 | model \| log |

### **STL Benchmarks on Wind Component (uv10)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU**. We provide config files in [configs/weather/simvp_uv10_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/simvp_uv10_5_625/) for `5.625` settings ($32\times 64$ resolutions). Notice that the input data of `uv10` has two channels.

| MetaFormer       |  Setting | Params | FLOPs |  FPS |   MSE  |   MAE  |  RMSE  |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:------:|:------:|:------:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   43 | 1.8976 | 0.9215 | 1.3775 | model \| log |
| PredRNN          | 50 epoch | 23.65M |  279G |   21 | 1.8810 | 0.9068 | 1.3715 | model \| log |
| PredRNN++        | 50 epoch | 38.40M |  414G |   14 | 1.8727 | 0.9019 | 1.3685 | model \| log |
| PredRNNv2        | 50 epoch | 23.68M |  280G |   21 | 2.0072 | 0.9413 | 1.4168 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.04G |  154 | 1.9993 | 0.9510 | 1.4140 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.02G |  498 | 1.5069 | 0.8142 | 1.2276 | model \| log |
| ViT              | 50 epoch | 12.42M |  8.0G |  427 | 1.6262 | 0.8438 | 1.2752 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.89G |  577 | 1.4996 | 0.8145 | 1.2246 | model \| log |
| Uniformer        | 50 epoch | 12.03M | 7.46G |  459 | 1.4850 | 0.8085 | 1.2186 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.93G |  707 | 1.6066 | 0.8395 | 1.2675 | model \| log |
| ConvMixer        | 50 epoch |  1.14M | 0.96G | 1698 | 1.7067 | 0.8714 | 1.3064 | model \| log |
| Poolformer       | 50 epoch |  9.99M | 5.62G |  717 | 1.6123 | 0.8410 | 1.2698 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.67G |  682 | 1.6914 | 0.8698 | 1.3006 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.71G |  520 | 1.5958 | 0.8371 | 1.2632 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.85G |  513 | 1.5539 | 0.8254 | 1.2466 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  411 | 1.6072 | 0.8451 | 1.2678 | model \| log |

### **STL Benchmarks on Cloud Cover (tcc)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU**. We provide config files in [configs/weather/simvp_tcc_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/simvp_tcc_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| MetaFormer       |  Setting | Params | FLOPs |  FPS |   MSE   |    MAE  |   RMSE  |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:-------:|:-------:|:-------:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 0.04944 | 0.15419 | 0.22234 | model \| log |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 0.05504 | 0.15877 | 0.23461 | model \| log |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 | 0.05479 | 0.15435 | 0.23407 | model \| log |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 | 0.05051 | 0.15867 | 0.22475 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 0.04765 | 0.15029 | 0.21829 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 0.04657 | 0.14688 | 0.21580 | model \| log |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 0.04778 | 0.15026 | 0.21859 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 0.04639 | 0.14729 | 0.21539 | model \| log |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 0.04680 | 0.14777 | 0.21634 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 0.04925 | 0.15264 | 0.22192 | model \| log |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 0.04717 | 0.14874 | 0.21718 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 0.04694 | 0.14884 | 0.21667 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 0.04742 | 0.14867 | 0.21775 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 0.04694 | 0.14725 | 0.21665 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 0.04692 | 0.14751 | 0.21661 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 0.04699 | 0.14802 | 0.21676 | model \| log |

<p align="right">(<a href="#top">back to top</a>)</p>
