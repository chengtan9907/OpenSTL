# Weather Prediction Benchmarks

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on the famous weather prediction datasets, WeatherBench. More STL methods will be supported in the future. Issues and PRs are welcome!** Currently, we only provide benchmark results, trained models and logs will be released soon (you can contact us if you require these files). You can download model files from [Baidu Cloud (brlk)](https://pan.baidu.com/s/1na73RW4I1sUme53WemnKeg?pwd=brlk).

## Table of Contents

- [WeatherBench Benchmarks](#weatherbench-benchmarks)
    - [STL/MetaVP Benchmarks on Temperature (t2m)](#stl-benchmarks-on-temperature-t2m)
    - [STL/MetaVP Benchmarks on Humidity (r)](#stl-benchmarks-on-humidity-r)
    - [STL/MetaVP Benchmarks on Wind Component (uv10)](#stl-benchmarks-on-wind-component-uv10)
    - [STL/MetaVP Benchmarks on Cloud Cover (tcc)](#stl-benchmarks-on-cloud-cover-tcc)
    - [STL/MetaVP Benchmarks on Multiple Variants (MV)](#stl-benchmarks-on-multiple-variants-mv)

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


## WeatherBench Benchmarks

We provide temperature prediction benchmark results on the popular [WeatherBench](https://arxiv.org/abs/2002.00469) dataset (temperature prediction `t2m`) using $12\rightarrow 12$ frames prediction setting. Metrics (MSE, MAE, SSIM, pSNR) of the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Cosine Annealing scheduler (no warmup and min lr is 1e-6).

### **STL Benchmarks on Temperature (t2m)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark STL methods and Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `t2m` (K). We provide config files in [configs/weather/t2m_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/t2m_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| Method           |  Setting | Params | FLOPs |  FPS |  MSE  |   MAE  |  RMSE |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:-----:|:------:|:-----:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 1.521 | 0.7949 | 1.233 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_convlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_convlstm_cos_ep50.log) |
| E3D-LSTM         | 50 epoch | 51.09M |  169G |   35 | 1.592 | 0.8059 | 1.262 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_e3dlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_e3dlstm_cos_ep50.log) |
| PhyDNet          | 50 epoch |  3.09M | 36.8G |  177 | 285.9 | 8.7370 | 16.91 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_phydnet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_phydnet_cos_ep50.log) |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 1.331 | 0.7246 | 1.154 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_predrnn_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_predrnn_cos_ep50.log) |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 | 1.634 | 0.7883 | 1.278 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_predrnnpp_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_predrnnpp_cos_ep50.log) |
| MIM              | 50 epoch | 37.75M |  109G |  126 | 1.784 | 0.8716 | 1.336 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_mim_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_mim_cos_ep50.log) |
| MAU              | 50 epoch |  5.46M | 39.6G |  237 | 1.251 | 0.7036 | 1.119 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_mau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_mau_cos_ep50.log) |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 | 1.545 | 0.7986 | 1.243 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_predrnnv2_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_predrnnv2_cos_ep50.log) |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 1.238 | 0.7037 | 1.113 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_incepu_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_incepu_cos_ep50.log) |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 1.105 | 0.6567 | 1.051 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_gsta_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_gsta_cos_ep50.log) |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 1.146 | 0.6712 | 1.070 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_vit_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_vit_cos_ep50.log) |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 1.143 | 0.6735 | 1.069 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_swin_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_swin_cos_ep50.log) |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 1.204 | 0.6885 | 1.097 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_uniformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_uniformer_cos_ep50.log) |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 1.255 | 0.7011 | 1.119 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_mlpmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_mlpmixer_cos_ep50.log) |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 1.267 | 0.7073 | 1.126 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_convmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_convmixer_cos_ep50.log) |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 1.156 | 0.6715 | 1.075 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_poolformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_poolformer_cos_ep50.log) |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 1.277 | 0.7220 | 1.130 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_convnext_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_convnext_cos_ep50.log) |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 1.150 | 0.6803 | 1.072 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_van_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_van_cos_ep50.log) |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 1.201 | 0.6906 | 1.096 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_hornet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_hornet_cos_ep50.log) |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 1.152 | 0.6665 | 1.073 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_moganet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_simvp_moganet_cos_ep50.log) |
| TAU              | 50 epoch | 12.22M | 6.70G |  511 | 1.162 | 0.6707 | 1.078 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_tau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_t2m_tau_cos_ep50.log) |

Then, we also provide the high-resolution benchmark of `t2m` using the similar training settings with **4GPUs** (4xbs4). The config files are in [configs/weather/t2m_1_40625](https://github.com/chengtan9907/OpenSTL/configs/weather/t2m_1_40625/) for `1.40625` settings ($128\times 256$ resolutions).

| Method           |  Setting | Params | FLOPs | FPS |  MSE  |   MAE  |   RMSE  |   Download   |
|------------------|:--------:|:------:|:-----:|:---:|:-----:|:------:|:-------:|:------------:|
| ConvLSTM         | 50 epoch | 15.08M |  550G |  35 | 1.0625 | 0.6517 |  1.031 | model \| log |
| PhyDNet          | 50 epoch |  3.09M |  148G |  41 | 297.34 | 8.9788 | 17.243 | model \| log |
| PredRNN          | 50 epoch | 23.84M | 1123G |   3 | 0.8966 | 0.5869 | 0.9469 | model \| log |
| PredRNN++        | 50 epoch | 38.58M | 1663G |   2 | 0.8538 | 0.5708 | 0.9240 | model \| log |
| MIM              | 50 epoch | 42.17M | 1739G |  11 | 1.2138 | 0.6857 | 1.1017 | model \| log |
| MAU              | 50 epoch | 11.82M |  172G |  17 | 1.0031 | 0.6316 | 1.0016 | model \| log |
| PredRNNv2        | 50 epoch | 23.86M | 1129G |   3 | 1.0451 | 0.6190 | 1.0223 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M |  128G |  27 | 0.8492 | 0.5636 | 0.9215 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M |  112G |  33 | 0.6499 | 0.4909 | 0.8062 | model \| log |
| ViT              | 50 epoch | 12.48M | 36.8G |  50 | 0.8969 | 0.5834 | 0.9470 | model \| log |
| Swin Transformer | 50 epoch | 12.42M |  110G |  38 | 0.7606 | 0.5193 | 0.8721 | model \| log |
| Uniformer        | 50 epoch | 12.09M | 48.8G |  57 | 1.0052 | 0.6294 | 1.0026 | model \| log |
| MLP-Mixer        | 50 epoch | 27.87M | 94.7G |  49 | 1.1865 | 0.6593 | 1.0893 | model \| log |
| ConvMixer        | 50 epoch |  1.14M | 15.1G | 117 | 0.8557 | 0.5669 | 0.9250 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 89.7G |  42 | 0.7983 | 0.5316 | 0.8935 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 90.5G |  47 | 0.8058 | 0.5406 | 0.8976 | model \| log |
| VAN              | 50 epoch | 12.15M |  107G |  34 | 0.7110 | 0.5094 | 0.8432 | model \| log |
| HorNet           | 50 epoch | 12.42M |  109G |  34 | 0.8250 | 0.5467 | 0.9083 | model \| log |
| MogaNet          | 50 epoch | 12.76M |  112G |  27 | 0.7517 | 0.5232 | 0.8670 | model \| log |
| TAU              | 50 epoch | 12.29M | 36.1G |  94 | 0.8316 | 0.5615 | 0.9119 | model \| log |

<p align="right">(<a href="#top">back to top</a>)</p>

### **STL Benchmarks on Humidity (r)**

Similar to [Weather Benchmark](weather_benchmarks.md#weatherBench-benchmarks), we benchmark STL methods and Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `r` (%). We provide config files in [configs/weather/r_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/r_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| Method           |  Setting | Params | FLOPs |  FPS |  MSE   |   MAE  |  RMSE |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:------:|:------:|:-----:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 35.146 |  4.012 | 5.928 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_convlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_convlstm_cos_ep50.log) |
| E3D-LSTM         | 50 epoch | 51.09M |  169G |   35 | 36.534 |  4.100 | 6.044 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_e3dlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_e3dlstm_cos_ep50.log) |
| PhyDNet          | 50 epoch |  3.09M | 36.8G |  177 | 239.00 |  8.975 | 15.46 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_phydnet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_phydnet_cos_ep50.log) |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 37.611 |  4.096 | 6.133 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_predrnn_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_predrnn_cos_ep50.log) |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 | 35.146 |  4.012 | 5.928 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_predrnnpp_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_predrnnpp_cos_ep50.log) |
| MIM              | 50 epoch | 37.75M |  109G |  126 | 36.534 |  4.100 | 6.044 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_mim_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_mim_cos_ep50.log) |
| MAU              | 50 epoch |  5.46M | 39.6G |  237 | 34.529 |  4.004 | 5.876 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_mau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_mau_cos_ep50.log) |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 | 36.508 |  4.087 | 6.042 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_predrnnv2_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_predrnnv2_cos_ep50.log) |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 34.355 |  3.994 | 5.861 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_incepu_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_incepu_cos_ep50.log) |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 31.426 |  3.765 | 5.606 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_gsta_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_gsta_cos_ep50.log) |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 32.616 |  3.852 | 5.711 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_vit_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_vit_cos_ep50.log) |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 31.332 |  3.776 | 5.597 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_swin_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_swin_cos_ep50.log) |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 32.199 |  3.864 | 5.674 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_uniformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_uniformer_cos_ep50.log) |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 34.467 |  3.950 | 5.871 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_mlpmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_mlpmixer_cos_ep50.log) |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 32.829 |  3.909 | 5.730 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_convmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_convmixer_cos_ep50.log) |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 31.989 |  3.803 | 5.656 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_poolformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_poolformer_cos_ep50.log) |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 33.179 |  3.928 | 5.760 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_convnext_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_convnext_cos_ep50.log) |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 31.712 |  3.812 | 5.631 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_van_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_van_cos_ep50.log) |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 32.081 |  3.826 | 5.664 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_hornet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_hornet_cos_ep50.log) |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 31.795 |  3.816 | 5.639 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_moganet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_simvp_moganet_cos_ep50.log) |
| TAU              | 50 epoch | 12.22M | 6.70G |  511 | 31.831 |  3.818 | 5.642 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_tau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_r_tau_cos_ep50.log) |

### **STL Benchmarks on Wind Component (uv10)**

Similar to [Weather Benchmark](weather_benchmarks.md#weatherBench-benchmarks), we benchmark STL methods and Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `uv10` (ms-1). We provide config files in [configs/weather/uv10_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/uv10_5_625/) for `5.625` settings ($32\times 64$ resolutions). Notice that the input data of `uv10` has two channels.

| Method           |  Setting | Params | FLOPs |  FPS |   MSE  |   MAE  |  RMSE  |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:------:|:------:|:------:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   43 | 1.8976 | 0.9215 | 1.3775 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_convlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_convlstm_cos_ep50.log) |
| E3D-LSTM         | 50 epoch | 51.81M |  171G |   35 | 2.4111 | 1.0342 | 1.5528 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_e3dlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_e3dlstm_cos_ep50.log) |
| PhyDNet          | 50 epoch |  3.09M | 36.8G |  172 | 16.798 | 2.9208 | 4.0986 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_phydnet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_phydnet_cos_ep50.log) |
| PredRNN          | 50 epoch | 23.65M |  279G |   21 | 1.8810 | 0.9068 | 1.3715 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_predrnn_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_predrnn_cos_ep50.log) |
| PredRNN++        | 50 epoch | 38.40M |  414G |   14 | 1.8727 | 0.9019 | 1.3685 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_predrnnpp_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_predrnnpp_cos_ep50.log) |
| MIM              | 50 epoch | 37.75M |  109G |  122 | 3.1399 | 1.1837 | 1.7720 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_mim_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_mim_cos_ep50.log) |
| MAU              | 50 epoch |  5.46M | 39.6G |  233 | 1.9001 | 0.9194 | 1.3784 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_mau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_mau_cos_ep50.log) |
| PredRNNv2        | 50 epoch | 23.68M |  280G |   21 | 2.0072 | 0.9413 | 1.4168 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_predrnnv2_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_predrnnv2_cos_ep50.log) |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.04G |  154 | 1.9993 | 0.9510 | 1.4140 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_incepu_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_incepu_cos_ep50.log) |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.02G |  498 | 1.5069 | 0.8142 | 1.2276 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_gsta_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_gsta_cos_ep50.log) |
| ViT              | 50 epoch | 12.42M |  8.0G |  427 | 1.6262 | 0.8438 | 1.2752 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_vit_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_vit_cos_ep50.log) |
| Swin Transformer | 50 epoch | 12.42M | 6.89G |  577 | 1.4996 | 0.8145 | 1.2246 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_swin_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_swin_cos_ep50.log) |
| Uniformer        | 50 epoch | 12.03M | 7.46G |  459 | 1.4850 | 0.8085 | 1.2186 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_uniformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_uniformer_cos_ep50.log) |
| MLP-Mixer        | 50 epoch | 11.10M | 5.93G |  707 | 1.6066 | 0.8395 | 1.2675 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_mlpmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_mlpmixer_cos_ep50.log) |
| ConvMixer        | 50 epoch |  1.14M | 0.96G | 1698 | 1.7067 | 0.8714 | 1.3064 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_convmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_convmixer_cos_ep50.log) |
| Poolformer       | 50 epoch |  9.99M | 5.62G |  717 | 1.6123 | 0.8410 | 1.2698 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_poolformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_poolformer_cos_ep50.log) |
| ConvNeXt         | 50 epoch | 10.09M | 5.67G |  682 | 1.6914 | 0.8698 | 1.3006 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_convnext_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_convnext_cos_ep50.log) |
| VAN              | 50 epoch | 12.15M | 6.71G |  520 | 1.5958 | 0.8371 | 1.2632 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_van_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_van_cos_ep50.log) |
| HorNet           | 50 epoch | 12.42M | 6.85G |  513 | 1.5539 | 0.8254 | 1.2466 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_hornet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_hornet_cos_ep50.log) |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  411 | 1.6072 | 0.8451 | 1.2678 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_moganet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_simvp_moganet_cos_ep50.log) |
| TAU              | 50 epoch | 12.22M | 6.70G |  505 | 1.5925 | 0.8426 | 1.2619 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_tau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_uv10_tau_cos_ep50.log) |

### **STL Benchmarks on Cloud Cover (tcc)**

Similar to [Weather Benchmark](weather_benchmarks.md#weatherBench-benchmarks), we benchmark STL methods and Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `tcc` (%). We provide config files in [configs/weather/tcc_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/tcc_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| Method           |  Setting | Params | FLOPs |  FPS |   MSE   |    MAE  |   RMSE  |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:-------:|:-------:|:-------:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 0.04944 | 0.15419 | 0.22234 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_convlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_convlstm_cos_ep50.log) |
| E3D-LSTM         | 50 epoch | 51.09M |  169G |   35 | 0.05729 | 0.15293 | 0.23936 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_e3dlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_e3dlstm_cos_ep50.log) |
| PhyDNet          | 50 epoch |  3.09M | 36.8G |  172 | 0.09913 | 0.22614 | 0.31485 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_phydnet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_phydnet_cos_ep50.log) |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 0.05504 | 0.15877 | 0.23461 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_predrnn_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_predrnn_cos_ep50.log) |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 | 0.05479 | 0.15435 | 0.23407 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_predrnnpp_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_predrnnpp_cos_ep50.log) |
| MIM              | 50 epoch | 37.75M |  109G |  126 | 0.05729 | 0.15293 | 0.23936 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_mim_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_mim_cos_ep50.log) |
| MAU              | 50 epoch |  5.46M | 39.6G |  237 | 0.04955 | 0.15158 | 0.22260 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_mau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_mau_cos_ep50.log) |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 | 0.05051 | 0.15867 | 0.22475 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_predrnnv2_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_predrnnv2_cos_ep50.log) |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 0.04765 | 0.15029 | 0.21829 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_incepu_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_incepu_cos_ep50.log) |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 0.04657 | 0.14688 | 0.21580 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_gsta_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_gsta_cos_ep50.log) |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 0.04778 | 0.15026 | 0.21859 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_vit_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_vit_cos_ep50.log) |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 0.04639 | 0.14729 | 0.21539 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_swin_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_swin_cos_ep50.log) |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 0.04680 | 0.14777 | 0.21634 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_uniformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_uniformer_cos_ep50.log) |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 0.04925 | 0.15264 | 0.22192 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_mlpmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_mlpmixer_cos_ep50.log) |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 0.04717 | 0.14874 | 0.21718 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_convmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_convmixer_cos_ep50.log) |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 0.04694 | 0.14884 | 0.21667 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_poolformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_poolformer_cos_ep50.log) |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 0.04742 | 0.14867 | 0.21775 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_convnext_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_convnext_cos_ep50.log) |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 0.04694 | 0.14725 | 0.21665 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_van_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_van_cos_ep50.log) |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 0.04692 | 0.14751 | 0.21661 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_hornet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_hornet_cos_ep50.log) |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 0.04699 | 0.14802 | 0.21676 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_moganet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_simvp_moganet_cos_ep50.log) |
| TAU              | 50 epoch | 12.22M | 6.70G |  511 | 0.04723 | 0.14604 | 0.21733 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_tau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/weather-5-625-weights/weather_tcc_tau_cos_ep50.log) |

<p align="right">(<a href="#top">back to top</a>)</p>

### **STL Benchmarks on Multiple Variants (MV)**

Using the similar setting as [Weather Benchmark](weather_benchmarks.md#weatherBench-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `r`, `t`, `u`, `v`, which have 13 levels and we chose 3 levels (150m, 500m, and 850m). We provide config files in [configs/weather/mv_4_s6_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/mv_4_s6_5_625/) for `5.625` settings ($32\times 64$ resolutions). Here, we employ Adam optimizer with Cosine Annealing scheduler (5-epoch warmup and min lr is 1e-6) to various methods. We provide results of each variant and the sum of four variants.

| Method (`sum`)   | Params |  FLOPs |  FPS | MSE(sum) | MAE(sum) | RMSE(sum)|   Download   |
|------------------|:------:|:------:|:----:|:--------:|:--------:|:--------:|:------------:|
| ConvLSTM         | 15.50M | 43.33G | 11   |  108.81  | 5.7439   |  8.1810  | model \| log |
| PhyDNet          |  3.10M | 11.25G | 14   |  228.83  | 10.109   |  13.213  | model \| log |
| PredRNN          | 24.56M | 88.02G | 5    |  104.16  | 5.5373   |  7.9553  | model \| log |
| PredRNN++        | 39.31M | 129.0G | 4    |  106.77  | 5.5821   |  8.0568  | model \| log |
| MIM              | 41.71M | 35.77G | 17   |  121.95  | 6.2786   |  8.7376  | model \| log |
| MAU              |  5.46M | 12.07G | 21   |  106.13  | 5.6487   |  7.9928  | model \| log |
| PredRNNv2        | 24.58M | 88.49G | 5    |  108.94  | 5.7747   |  8.1872  | model \| log |
| IncepU (SimVPv1) | 13.80M |  7.26G | 16   |  108.50  | 5.7360   |  8.1165  | model \| log |
| gSTA (SimVPv2)   |  9.96M |  5.25G | 18   |  103.36  | 5.4856   |  7.9059  | model \| log |
| ViT              |  9.66M |  6.12G | 8    |  102.90  | 5.4776   |  7.8643  | model \| log |
| Swin Transformer |  9.66M |  5.15G | 24   |  102.14  | 5.4400   |  7.8325  | model \| log |
| Uniformer        |  9.53M |  5.85G | 9    |  102.39  | 5.4361   |  7.8225  | model \| log |
| MLP-Mixer        |  8.74M |  4.39G | 32   |  107.65  | 5.6820   |  8.1058  | model \| log |
| ConvMixer        |  0.85M |  0.49G | 157  |  112.76  | 5.9114   |  8.3238  | model \| log |
| Poolformer       |  7.76M |  4.14G | 26   |  114.05  | 5.9979   |  8.4760  | model \| log |
| ConvNeXt         |  7.85M |  4.19G | 42   |  108.66  | 5.7432   |  8.1763  | model \| log |
| VAN              |  9.49M |  5.01G | 21   |  99.816  | 5.3351   |  7.7075  | model \| log |
| HorNet           |  9.68M |  5.12G | 21   |  104.21  | 5.5181   |  7.9854  | model \| log |
| MogaNet          |  9.97M |  5.25G | 17   |  98.664  | 5.3003   |  7.6539  | model \| log |
| TAU              |  9.55M |  5.01G | 21   |  99.428  | 5.3282   |  7.6855  | model \| log |

| Method (`r`)     | Params |  FLOPs |  FPS |  MSE(r)  |  MAE(r)  |  RMSE(r) |   Download   |
|------------------|:------:|:------:|:----:|:--------:|:--------:|:--------:|:------------:|
| ConvLSTM         | 15.50M | 43.33G | 11   |  368.15  | 13.490   |  19.187  | model \| log |
| PhyDNet          |  3.10M | 11.25G | 14   |  668.40  | 21.398   |  25.853  | model \| log |
| PredRNN          | 24.56M | 88.02G | 5    |  354.57  | 13.169   |  18.830  | model \| log |
| PredRNN++        | 39.31M | 129.0G | 4    |  363.15  | 13.246   |  19.056  | model \| log |
| MIM              | 41.71M | 35.77G | 17   |  408.24  | 14.658   |  20.205  | model \| log |
| MAU              |  5.46M | 12.07G | 21   |  363.36  | 13.503   |  19.062  | model \| log |
| PredRNNv2        | 24.58M | 88.49G | 5    |  368.52  | 13.594   |  19.197  | model \| log |
| IncepU (SimVPv1) | 13.80M |  7.26G | 16   |  370.03  | 13.584   |  19.236  | model \| log |
| gSTA (SimVPv2)   |  9.96M |  5.25G | 18   |  352.79  | 13.021   |  18.783  | model \| log |
| ViT              |  9.66M |  6.12G | 8    |  352.36  | 13.056   |  18.771  | model \| log |
| Swin Transformer |  9.66M |  5.15G | 24   |  349.92  | 12.984   |  18.706  | model \| log |
| Uniformer        |  9.53M |  5.85G | 9    |  351.66  | 12.994   |  18.753  | model \| log |
| MLP-Mixer        |  8.74M |  4.39G | 32   |  365.48  | 13.408   |  19.118  | model \| log |
| ConvMixer        |  0.85M |  0.49G | 157  |  381.85  | 13.917   |  19.541  | model \| log |
| Poolformer       |  7.76M |  4.14G | 26   |  380.18  | 13.908   |  19.498  | model \| log |
| ConvNeXt         |  7.85M |  4.19G | 42   |  367.39  | 13.516   |  19.168  | model \| log |
| VAN              |  9.49M |  5.01G | 21   |  343.61  | 12.790   |  18.537  | model \| log |
| HorNet           |  9.68M |  5.12G | 21   |  353.02  | 13.024   |  18.789  | model \| log |
| MogaNet          |  9.97M |  5.25G | 17   |  340.06  | 12.738   |  18.441  | model \| log |
| TAU              |  9.55M |  5.01G | 21   |  342.63  | 12.801   |  18.510  | model \| log |

| Method (`t`)     | Params |  FLOPs |  FPS |  MSE(t)  |  MAE(t)  |  RMSE(t) |   Download   |
|------------------|:------:|:------:|:----:|:--------:|:--------:|:--------:|:------------:|
| ConvLSTM         | 15.50M | 43.33G | 11   |  6.3034  | 1.7695   |  2.5107  | model \| log |
| PhyDNet          |  3.10M | 11.25G | 14   |  95.113  | 6.4749   |  9.7526  | model \| log |
| PredRNN          | 24.56M | 88.02G | 5    |  5.5966  | 1.6411   |  2.3657  | model \| log |
| PredRNN++        | 39.31M | 129.0G | 4    |  5.6471  | 1.6433   |  2.3763  | model \| log |
| MIM              | 41.71M | 35.77G | 17   |  7.5152  | 1.9650   |  2.7414  | model \| log |
| MAU              |  5.46M | 12.07G | 21   |  5.6287  | 1.6810   |  2.3725  | model \| log |
| PredRNNv2        | 24.58M | 88.49G | 5    |  6.3078  | 1.7770   |  2.5110  | model \| log |
| IncepU (SimVPv1) | 13.80M |  7.26G | 16   |  6.1068  | 1.7554   |  2.4712  | model \| log |
| gSTA (SimVPv2)   |  9.96M |  5.25G | 18   |  5.4382  | 1.6129   |  2.3319  | model \| log |
| ViT              |  9.66M |  6.12G | 8    |  5.2722  | 1.6005   |  2.2961  | model \| log |
| Swin Transformer |  9.66M |  5.15G | 24   |  5.2486  | 1.5856   |  2.2910  | model \| log |
| Uniformer        |  9.53M |  5.85G | 9    |  5.1174  | 1.5758   |  2.2622  | model \| log |
| MLP-Mixer        |  8.74M |  4.39G | 32   |  5.8546  | 1.6948   |  2.4196  | model \| log |
| ConvMixer        |  0.85M |  0.49G | 157  |  6.5838  | 1.8228   |  2.5660  | model \| log |
| Poolformer       |  7.76M |  4.14G | 26   |  7.1077  | 1.8791   |  2.6660  | model \| log |
| ConvNeXt         |  7.85M |  4.19G | 42   |  6.1749  | 1.7448   |  2.4849  | model \| log |
| VAN              |  9.49M |  5.01G | 21   |  4.9396  | 1.5390   |  2.2225  | model \| log |
| HorNet           |  9.68M |  5.12G | 21   |  5.5856  | 1.6198   |  2.3634  | model \| log |
| MogaNet          |  9.97M |  5.25G | 17   |  4.8335  | 1.5246   |  2.1985  | model \| log |
| TAU              |  9.55M |  5.01G | 21   |  4.9042  | 1.5341   |  2.2145  | model \| log |

| Method (`u`)     | Params |  FLOPs |  FPS |  MSE(u)  |  MAE(u)  |  RMSE(u) |   Download   |
|------------------|:------:|:------:|:----:|:--------:|:--------:|:--------:|:------------:|
| ConvLSTM         | 15.50M | 43.33G | 11   |  30.002  | 3.8923   |  5.4774  | model \| log |
| PhyDNet          |  3.10M | 11.25G | 14   |  97.424  | 7.3637   |  9.8704  | model \| log |
| PredRNN          | 24.56M | 88.02G | 5    |  27.484  | 3.6776   |  5.2425  | model \| log |
| PredRNN++        | 39.31M | 129.0G | 4    |  28.396  | 3.7322   |  5.3288  | model \| log |
| MIM              | 41.71M | 35.77G | 17   |  35.586  | 4.2842   |  5.9654  | model \| log |
| MAU              |  5.46M | 12.07G | 21   |  27.582  | 3.7409   |  5.2519  | model \| log |
| PredRNNv2        | 24.58M | 88.49G | 5    |  29.833  | 3.8870   |  5.4620  | model \| log |
| IncepU (SimVPv1) | 13.80M |  7.26G | 16   |  28.782  | 3.8435   |  5.3649  | model \| log |
| gSTA (SimVPv2)   |  9.96M |  5.25G | 18   |  27.166  | 3.6747   |  5.2121  | model \| log |
| ViT              |  9.66M |  6.12G | 8    |  26.595  | 3.6472   |  5.1570  | model \| log |
| Swin Transformer |  9.66M |  5.15G | 24   |  26.292  | 3.6133   |  5.1276  | model \| log |
| Uniformer        |  9.53M |  5.85G | 9    |  25.994  | 3.6069   |  5.0985  | model \| log |
| MLP-Mixer        |  8.74M |  4.39G | 32   |  29.242  | 3.8407   |  5.4076  | model \| log |
| ConvMixer        |  0.85M |  0.49G | 157  |  30.983  | 3.9949   |  5.5662  | model \| log |
| Poolformer       |  7.76M |  4.14G | 26   |  33.757  | 4.1280   |  5.8101  | model \| log |
| ConvNeXt         |  7.85M |  4.19G | 42   |  29.764  | 3.8688   |  5.4556  | model \| log |
| VAN              |  9.49M |  5.01G | 21   |  24.991  | 3.5254   |  4.9991  | model \| log |
| HorNet           |  9.68M |  5.12G | 21   |  28.192  | 3.7142   |  5.3096  | model \| log |
| MogaNet          |  9.97M |  5.25G | 17   |  24.535  | 3.4882   |  4.9533  | model \| log |
| TAU              |  9.55M |  5.01G | 21   |  24.719  | 3.5060   |  4.9719  | model \| log |

| Method (`v`)     | Params |  FLOPs |  FPS |  MSE(v)  |  MAE(v)  |  RMSE(v) |   Download   |
|------------------|:------:|:------:|:----:|:--------:|:--------:|:--------:|:------------:|
| ConvLSTM         | 15.50M | 43.33G | 11   |  30.789  | 3.8238   |  5.5488  | model \| log |
| PhyDNet          |  3.10M | 11.25G | 14   |  54.389  | 5.1996   |  7.3749  | model \| log |
| PredRNN          | 24.56M | 88.02G | 5    |  28.973  | 3.6617   |  5.3827  | model \| log |
| PredRNN++        | 39.31M | 129.0G | 4    |  29.872  | 3.7067   |  5.4655  | model \| log |
| MIM              | 41.71M | 35.77G | 17   |  36.464  | 4.2066   |  6.0386  | model \| log |
| MAU              |  5.46M | 12.07G | 21   |  27.929  | 3.6700   |  5.2848  | model \| log |
| PredRNNv2        | 24.58M | 88.49G | 5    |  31.119  | 3.8406   |  5.5785  | model \| log |
| IncepU (SimVPv1) | 13.80M |  7.26G | 16   |  29.094  | 3.7614   |  5.3939  | model \| log |
| gSTA (SimVPv2)   |  9.96M |  5.25G | 18   |  28.058  | 3.6335   |  5.2970  | model \| log |
| ViT              |  9.66M |  6.12G | 8    |  27.381  | 3.6068   |  5.2327  | model \| log |
| Swin Transformer |  9.66M |  5.15G | 24   |  27.097  | 3.5777   |  5.2055  | model \| log |
| Uniformer        |  9.53M |  5.85G | 9    |  26.799  | 3.5676   |  5.1768  | model \| log |
| MLP-Mixer        |  8.74M |  4.39G | 32   |  30.014  | 3.7840   |  5.4785  | model \| log |
| ConvMixer        |  0.85M |  0.49G | 157  |  31.609  | 3.9104   |  5.6222  | model \| log |
| Poolformer       |  7.76M |  4.14G | 26   |  35.161  | 4.0764   |  5.9296  | model \| log |
| ConvNeXt         |  7.85M |  4.19G | 42   |  31.326  | 3.8435   |  5.5969  | model \| log |
| VAN              |  9.49M |  5.01G | 21   |  25.720  | 3.4858   |  5.0715  | model \| log |
| HorNet           |  9.68M |  5.12G | 21   |  30.028  | 3.7148   |  5.4798  | model \| log |
| MogaNet          |  9.97M |  5.25G | 17   |  25.232  | 3.4509   |  5.0231  | model \| log |
| TAU              |  9.55M |  5.01G | 21   |  25.456  | 3.4723   |  5.0454  | model \| log |

<p align="right">(<a href="#top">back to top</a>)</p>
