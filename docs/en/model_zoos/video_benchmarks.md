# Video Prediction Benchmarks

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on various video prediction datasets. More STL methods will be supported in the future. Issues and PRs are welcome!** Currently, we only provide benchmark results, trained models and logs will be released soon (you can contact us if you require these files).

<details open>
<summary>Currently supported spatiotemporal prediction methods</summary>

- [x] [ConvLSTM](https://arxiv.org/abs/1506.04214) (NeurIPS'2015)
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


## Moving MNIST Benchmarks

We provide benchmark results on the popular [Moving MNIST](http://arxiv.org/abs/1502.04681) dataset using $10\rightarrow 10$ frames prediction setting following [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855). Metrics (MSE, MAE, SSIM, pSNR) of the the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Onecycle scheduler and **single GPU**.

### **STL Benchmarks on MMNIST**

For a fair comparison of different methods, we report the best results when models are trained to convergence. We provide config files in [configs/mmnist](https://github.com/chengtan9907/OpenSTL/configs/mmnist).

| Method       |   Setting  | Params |  FLOPs | FPS |  MSE  |  MAE  |  SSIM  |  PSNR |   Download   |
|--------------|:----------:|:------:|:------:|:---:|:-----:|:-----:|:------:|:-----:|:------------:|
| ConvLSTM-S   |  200 epoch |  15.0M |  56.8G | 113 | 29.80 | 90.64 | 0.9288 | 37.36 | model \| log |
| ConvLSTM-L   |  200 epoch |  33.8M | 127.0G |  50 | 27.78 | 86.14 | 0.9343 | 37.46 | model \| log |
| PhyDNet      |  200 epoch |   3.1M |  15.3G | 182 | 28.19 | 78.64 | 0.9374 | 38.11 | model \| log |
| PredRNN      |  200 epoch |  23.8M | 116.0G |  54 | 23.97 | 72.82 | 0.9462 | 38.20 | model \| log |
| PredRNN++    |  200 epoch |  38.6M | 171.7G |  38 | 22.06 | 69.58 | 0.9509 | 38.30 | model \| log |
| MIM          |  200 epoch |  38.0M | 179.2G |  37 | 22.55 | 69.97 | 0.9498 | 38.16 | model \| log |
| MAU          |  200 epoch |   4.5M |  17.8G | 201 | 26.86 | 78.22 | 0.9398 | 38.05 | model \| log |
| E3D-LSTM     |  200 epoch |  51.0M | 298.9G |  18 | 35.97 | 78.28 | 0.9320 | 39.38 | model \| log |
| CrevNet      |  200 epoch |   5.0M | 270.7G |  10 | 30.15 | 86.28 | 0.9350 |       | model \| log |
| PredRNN.V2   |  200 epoch |  23.9M | 116.6G |  52 | 24.13 | 73.73 | 0.9453 | 38.17 | model \| log |
| SimVP+IncepU |  200 epoch |  58.0M |  19.4G | 209 | 32.15 | 89.05 | 0.9268 | 37.97 | model \| log |
| SimVP+gSTA-S |  200 epoch |  46.8M |  16.5G | 282 | 26.69 | 77.19 | 0.9402 | 38.35 | model \| log |
| SimVP+gSTA-S |  200 epoch |  44.7M |  16.0G | 283 | 24.60 | 71.93 | 0.9454 | 38.51 | model \| log |

### **Benchmark of MetaFormers Based on SimVP**

Since the hidden Translator in [SimVP](https://arxiv.org/abs/2211.12509) can be replaced by any [Metaformer](https://arxiv.org/abs/2111.11418) block which achieves `token mixing` and `channel mixing`, we benchmark popular Metaformer architectures on SimVP with training times of 200-epoch and 2000-epoch. We provide config file in [configs/mmnist/simvp](https://github.com/chengtan9907/OpenSTL/configs/mmnist/simvp/).

| MetaFormer       |   Setting  | Params |  FLOPs | FPS |  MSE  |  MAE  |  SSIM  |  PSNR |   Download   |
|------------------|:----------:|:------:|:------:|:---:|:-----:|:-----:|:------:|:-----:|:------------:|
| IncepU (SimVPv1) |  200 epoch |  58.0M |  19.4G | 209 | 32.15 | 89.05 | 0.9268 | 37.97 | model \| log |
| gSTA (SimVPv2)   |  200 epoch |  46.8M |  16.5G | 282 | 26.69 | 77.19 | 0.9402 | 38.35 | model \| log |
| ViT              |  200 epoch |  46.1M |  16.9G | 290 | 35.15 | 95.87 | 0.9139 | 37.79 | model \| log |
| Swin Transformer |  200 epoch |  46.1M |  16.4G | 294 | 29.70 | 84.05 | 0.9331 | 38.14 | model \| log |
| Uniformer        |  200 epoch |  44.8M |  16.5G | 296 | 30.38 | 85.87 | 0.9308 | 38.11 | model \| log |
| MLP-Mixer        |  200 epoch |  38.2M |  14.7G | 334 | 29.52 | 83.36 | 0.9338 | 38.19 | model \| log |
| ConvMixer        |  200 epoch |   3.9M |   5.5G | 658 | 32.09 | 88.93 | 0.9259 | 37.97 | model \| log |
| Poolformer       |  200 epoch |  37.1M |  14.1G | 341 | 31.79 | 88.48 | 0.9271 | 38.06 | model \| log |
| ConvNeXt         |  200 epoch |  37.3M |  14.1G | 344 | 26.94 | 77.23 | 0.9397 | 38.34 | model \| log |
| VAN              |  200 epoch |  44.5M |  16.0G | 288 | 26.10 | 76.11 | 0.9417 | 38.39 | model \| log |
| HorNet           |  200 epoch |  45.7M |  16.3G | 287 | 29.64 | 83.26 | 0.9331 | 38.16 | model \| log |
| MogaNet          |  200 epoch |  46.8M |  16.5G | 255 | 25.57 | 75.19 | 0.9429 | 38.41 | model \| log |
| IncepU (SimVPv1) | 2000 epoch |  58.0M |  19.4G | 209 | 21.15 | 64.15 | 0.9536 | 38.81 | model \| log |
| gSTA (SimVPv2)   | 2000 epoch |  46.8M |  16.5G | 282 | 15.05 | 49.80 | 0.9670 |   -   | model \| log |
| ViT              | 2000 epoch |  46.1M | 16.9.G | 290 | 19.74 | 61.65 | 0.9539 | 38.96 | model \| log |
| Swin Transformer | 2000 epoch |  46.1M |  16.4G | 294 | 19.11 | 59.84 | 0.9584 | 39.03 | model \| log |
| Uniformer        | 2000 epoch |  44.8M |  16.5G | 296 | 18.01 | 57.52 | 0.9609 | 39.11 | model \| log |
| MLP-Mixer        | 2000 epoch |  38.2M |  14.7G | 334 | 18.85 | 59.86 | 0.9589 | 38.98 | model \| log |
| ConvMixer        | 2000 epoch |   3.9M |   5.5G | 658 | 22.30 | 67.37 | 0.9507 | 38.67 | model \| log |
| Poolformer       | 2000 epoch |  37.1M |  14.1G | 341 | 20.96 | 64.31 | 0.9539 | 38.86 | model \| log |
| ConvNeXt         | 2000 epoch |  37.3M |  14.1G | 344 | 17.58 | 55.76 | 0.9617 | 39.19 | model \| log |
| VAN              | 2000 epoch |  44.5M |  16.0G | 288 | 16.21 | 53.57 | 0.9646 | 39.26 | model \| log |
| HorNet           | 2000 epoch |  45.7M |  16.3G | 287 | 17.40 | 55.70 | 0.9624 | 39.19 | model \| log |
| MogaNet          | 2000 epoch |  46.8M |  16.5G | 255 | 15.67 | 51.84 | 0.9661 | 39.35 | model \| log |

<p align="right">(<a href="#top">back to top</a>)</p>

## Moving FMNIST Benchmarks

Similar to [Moving MNIST](http://arxiv.org/abs/1502.04681), we also provide the advanced version of MNIST, i.e., MFMNIST benchmark results, using $10\rightarrow 10$ frames prediction setting following [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855). Metrics (MSE, MAE, SSIM, pSNR) of the the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Onecycle scheduler and **single GPU**.

### **STL Benchmarks on MFMNIST**

For a fair comparison of different methods, we report the best results when models are trained to convergence. We provide config files in [configs/mfmnist](https://github.com/chengtan9907/OpenSTL/configs/mfmnist).

| Method       |   Setting  | Params |  FLOPs | FPS |  MSE  |   MAE  |  SSIM  |  PSNR |   Download   |
|--------------|:----------:|:------:|:------:|:---:|:-----:|:------:|:------:|:-----:|:------------:|
| ConvLSTM-S   |  200 epoch |  15.0M |  56.8G | 113 | 28.87 | 113.20 | 0.8793 | 35.09 | model \| log |
| ConvLSTM-L   |  200 epoch |  33.8M | 127.0G |  50 | 25.51 | 104.85 | 0.8928 | 35.23 | model \| log |
| PhyDNet      |  200 epoch |  3.1M  |  15.3G | 182 | | | | | model \| log |
| PredRNN      |  200 epoch |  23.8M | 116.0G |  54 | 22.01 |  91.74 | 0.9091 | 35.74 | model \| log |
| PredRNN++    |  200 epoch |  38.6M | 171.7G |  38 | | | | | model \| log |
| MIM          |  200 epoch |  38.0M | 179.2G |  37 | 23.09 |  96.37 | 0.9043 | 35.53 | model \| log |
| MIM          |  200 epoch |   4.5M |  17.8G | 201 | 26.56 | 104.39 | 0.8916 | 35.41 | model \| log |
| E3D-LSTM     |  200 epoch |  51.0M | 298.9G |  18 | | | | | model \| log |
| CrevNet      |  200 epoch |   5.0M | 270.7G |  10 | | | | | model \| log |
| PredRNN.V2   |  200 epoch |  23.9M | 116.6G |  52 | 24.13 |  97.46 | 0.9004 | 35.60 | model \| log |
| SimVP+IncepU |  200 epoch |  58.0M |  19.4G | 209 | 30.77 | 113.94 | 0.8740 | 35.31 | model \| log |
| SimVP+gSTA-S |  200 epoch |  46.8M |  16.5G | 282 | 25.86 | 101.22 | 0.8933 | 35.58 | model \| log |

### **Benchmark of MetaFormers Based on SimVP**

Since the hidden Translator in [SimVP](https://arxiv.org/abs/2211.12509) can be replaced by any [Metaformer](https://arxiv.org/abs/2111.11418) block which achieves `token mixing` and `channel mixing`, we benchmark popular Metaformer architectures on SimVP with training times of 200 epochs. We provide config file in [configs/mfmnist/simvp](https://github.com/chengtan9907/OpenSTL/configs/mfmnist/simvp/).

| MetaFormer       |   Setting  | Params |  FLOPs | FPS |  MSE  |   MAE  |  SSIM  |  PSNR |   Download   |
|------------------|:----------:|:------:|:------:|:---:|:-----:|:------:|:------:|:-----:|:------------:|
| IncepU (SimVPv1) |  200 epoch |  58.0M |  19.4G | 209 | 30.77 | 113.94 | 0.8740 | 35.31 | model \| log |
| gSTA (SimVPv2)   |  200 epoch |  46.8M |  16.5G | 282 | 25.86 | 101.22 | 0.8933 | 35.58 | model \| log |
| ViT              |  200 epoch |  46.1M | 16.9.G | 290 | 31.05 | 115.59 | 0.8712 | 35.30 | model \| log |
| Swin Transformer |  200 epoch |  46.1M |  16.4G | 294 | 28.66 | 108.93 | 0.8815 | 35.43 | model \| log |
| Uniformer        |  200 epoch |  44.8M |  16.5G | 296 | 29.56 | 111.72 | 0.8779 | 35.36 | model \| log |
| MLP-Mixer        |  200 epoch |  38.2M |  14.7G | 334 | 28.83 | 109.51 | 0.8803 | 35.40 | model \| log |
| ConvMixer        |  200 epoch |   3.9M |   5.5G | 658 | 31.21 | 115.74 | 0.8709 | 35.26 | model \| log |
| Poolformer       |  200 epoch |  37.1M |  14.1G | 341 | 30.02 | 113.07 | 0.8750 | 35.35 | model \| log |
| ConvNeXt         |  200 epoch |  37.3M |  14.1G | 344 | 26.41 | 102.56 | 0.8908 | 35.55 | model \| log |
| VAN              |  200 epoch |  44.5M |  16.0G | 288 | 31.39 | 116.28 | 0.8703 | 35.26 | model \| log |
| HorNet           |  200 epoch |  45.7M |  16.3G | 287 | 29.19 | 110.17 | 0.8796 | 35.39 | model \| log |
| MogaNet          |  200 epoch |  46.8M |  16.5G | 255 | 25.14 |  99.69 | 0.8960 | 35.62 | model \| log |

<p align="right">(<a href="#top">back to top</a>)</p>

## KittiCaltech Benchmarks

We provide benchmark results on [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) dataset using $10\rightarrow 1$ frames prediction setting following [PredNet](https://arxiv.org/abs/1605.08104). Metrics (MSE, MAE, SSIM, pSNR) of the the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. The default training setup is trained 100 epochs by Adam optimizer with Onecycle scheduler on **single GPU**, while some computational consuming methods (denoted by \*) using **4GPUs**.

### **STL Benchmarks on KittiCaltech**

For a fair comparison of different methods, we report the best results when models are trained to convergence. We provide config files in [configs/kitticaltech](https://github.com/chengtan9907/OpenSTL/configs/kitticaltech).

| Method       |  Setting  | Params |  FLOPs | FPS |  MSE  |   MAE  |  SSIM  |  PSNR |   Download   |
|--------------|:---------:|:------:|:------:|:---:|:-----:|:------:|:------:|:-----:|:------------:|
| ConvLSTM-S   | 100 epoch |  15.0M | 595.0G |  33 | 139.6 | 1583.3 | 0.9345 | 32.82 | model \| log |
| E3D-LSTM\*   | 100 epoch |  54.9M |  1004G |  10 | 203.7 | 1929.7 | 0.9062 | 32.04 | model \| log |
| MAU          | 100 epoch |  24.3M | 172.0G |  16 | 177.8 | 1800.4 | 0.9176 | 32.24 | model \| log |
| MIM          | 100 epoch |  49.2M |  1858G |  39 | 127.3 | 1461.1 | 0.9410 | 33.26 | model \| log |
| PredRNN      | 100 epoch |  23.7M |  1216G |  17 | 130.4 | 1525.5 | 0.9374 | 33.01 | model \| log |
| PredRNN++    | 100 epoch |  38.5M |  1803G |  12 | 125.5 | 1453.2 | 0.9433 | 33.27 | model \| log |
| PredRNN.V2   | 100 epoch |  23.8M |  1223G |  52 | 147.8 | 1610.5 | 0.9330 | 32.67 | model \| log |
| SimVP+IncepU | 100 epoch |   8.6M |  60.6G |  57 | 160.2 | 1690.8 | 0.9338 | 32.48 | model \| log |
| SimVP+gSTA-S | 100 epoch |  15.6M |  96.3G |  40 | 129.7 | 1507.7 | 0.9454 | 33.05 | model \| log |

### **Benchmark of MetaFormers Based on SimVP**

Since the hidden Translator in [SimVP](https://arxiv.org/abs/2211.12509) can be replaced by any [Metaformer](https://arxiv.org/abs/2111.11418) block which achieves `token mixing` and `channel mixing`, we benchmark popular Metaformer architectures on SimVP with 100-epoch training. We provide config file in [configs/kitticaltech/simvp](https://github.com/chengtan9907/OpenSTL/configs/kitticaltech/simvp/).

| MetaFormer       |  Setting  | Params |  FLOPs | FPS |  MSE  |  MAE   |  SSIM  |  PSNR |   Download   |
|------------------|:---------:|:------:|:------:|:---:|:-----:|:------:|:------:|:-----:|:------------:|
| IncepU (SimVPv1) | 100 epoch |   8.6M |  60.6G |  57 | 160.2 | 1690.8 | 0.9338 | 32.48 | model \| log |
| gSTA (SimVPv2)   | 100 epoch |  15.6M |  96.3G |  40 | 129.7 | 1507.7 | 0.9454 | 33.05 | model \| log |
| ViT\*            | 100 epoch |  12.7M | 155.0G |  25 | 146.4 | 1615.8 | 0.9379 | 32.58 | model \| log |
| Swin Transformer | 100 epoch |  15.3M |  95.2G |  49 | 155.2 | 1588.9 | 0.9299 | 32.98 | model \| log |
| Uniformer\*      | 100 epoch |  11.8M | 104.0G |  28 | 135.9 | 1534.2 | 0.9393 | 32.94 | model \| log |
| MLP-Mixer        | 100 epoch |  22.2M |  83.5G |  60 | 207.9 | 1835.9 | 0.9133 | 32.37 | model \| log |
| ConvMixer        | 100 epoch |   1.5M |  23.1G | 129 | 174.7 | 1854.3 | 0.9232 | 31.88 | model \| log |
| Poolformer       | 100 epoch |  12.4M |  79.8G |  51 | 153.4 | 1613.5 | 0.9334 | 32.79 | model \| log |
| ConvNeXt         | 100 epoch |  12.5M |  80.2G |  54 | 146.8 | 1630.0 | 0.9336 | 32.58 | model \| log |
| VAN              | 100 epoch |  14.9M |  92.5G |  41 | 132.1 | 1501.5 | 0.9437 | 33.10 | model \| log |
| HorNet           | 100 epoch |  15.3M |  94.4G |  43 | 152.8 | 1637.9 | 0.9365 | 32.70 | model \| log |
| MogaNet          | 100 epoch |  15.6M |  96.2G |  36 | 131.4 | 1512.1 | 0.9442 | 32.93 | model \| log |

<p align="right">(<a href="#top">back to top</a>)</p>
