# Traffic Prediction Benchmarks

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on popular traffic prediction datasets. More STL methods will be supported in the future. Issues and PRs are welcome!** Currently, we only provide benchmark results, trained models and logs will be released soon (you can contact us if you require these files). You can download model files from [Baidu Cloud (3t2t)](https://pan.baidu.com/s/1dH3gS9pyl3SQP8mL2FBgoA?pwd=3t2t).

## Table of Contents

- [TaxiBJ Benchmarks](#taxibj-benchmarks)
    - [STL Benchmarks on TaxiBJ](#stl-benchmarks-on-taxibj)
    - [MetaVP Benchmarks on TaxiBJ](#benchmark-of-metaformers-on-simvp-metavp)

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


## TaxiBJ Benchmarks

We provide traffic benchmark results on the popular [TaxiBJ](https://arxiv.org/abs/1610.00081) dataset using $4\rightarrow 4$ frames prediction setting. Metrics (MSE, MAE, SSIM, pSNR) of the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Cosine Annealing scheduler (5 epochs warmup and min lr is 1e-6) and **single GPU**.

### **STL Benchmarks on TaxiBJ**

For a fair comparison of different methods, we report the best results when models are trained to convergence. We provide config files in [configs/taxibj](https://github.com/chengtan9907/OpenSTL/configs/taxibj).

| Method       |  Setting | Params |  FLOPs |  FPS |   MSE  |  MAE  |  SSIM  |  PSNR |   Download   |
|--------------|:--------:|:------:|:------:|:----:|:------:|:-----:|:------:|:-----:|:------------:|
| ConvLSTM-S   | 50 epoch | 14.98M | 20.74G |  815 | 0.3358 | 15.32 | 0.9836 | 39.45 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_convlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_convlstm_cos_ep50.log) |
| E3D-LSTM\*   | 50 epoch | 50.99M | 98.19G |   60 | 0.3427 | 14.98 | 0.9842 | 39.64 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_e3dlstm_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_e3dlstm_cos_ep50.log) |
| PhyDNet      | 50 epoch |  3.09M |  5.60G |  982 | 0.3622 | 15.53 | 0.9828 | 39.46 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_phydnet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_phydnet_cos_ep50.log) |
| PredNet      | 50 epoch | 12.5M  |  0.85G | 5031 | 0.3516 | 15.91 | 0.9828 | 39.29 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_prednet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_prednet_cos_ep50.log) |
| PredRNN      | 50 epoch | 23.66M | 42.40G |  416 | 0.3194 | 15.31 | 0.9838 | 39.51 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_predrnn_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_predrnn_cos_ep50.log) |
| MIM          | 50 epoch | 37.86M | 64.10G |  275 | 0.3110 | 14.96 | 0.9847 | 39.65 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_mim_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_mim_cos_ep50.log) |
| MAU          | 50 epoch |  4.41M |  6.02G |  540 | 0.3268 | 15.26 | 0.9834 | 39.52 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_mau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_mau_cos_ep50.log) |
| PredRNN++    | 50 epoch | 38.40M | 62.95G |  301 | 0.3348 | 15.37 | 0.9834 | 39.47 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_predrnnpp_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_predrnnpp_cos_ep50.log) |
| PredRNN.V2   | 50 epoch | 23.67M | 42.63G |  378 | 0.3834 | 15.55 | 0.9826 | 39.49 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_predrnnv2_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_predrnnv2_cos_ep50.log) |
| DMVFN        | 50 epoch |  3.54M | 0.057G | 6347 | 3.3954 | 45.52 | 0.8321 | 31.14 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_dmvfn_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_dmvfn_cos_ep50.log) |
| SimVP+IncepU | 50 epoch | 13.79M |  3.61G |  533 | 0.3282 | 15.45 | 0.9835 | 39.45 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_incepu_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_incepu_cos_ep50.log) |
| SimVP+gSTA-S | 50 epoch |  9.96M |  2.62G | 1217 | 0.3246 | 15.03 | 0.9844 | 39.71 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_gsta_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_gsta_cos_ep50.log) |
| TAU          | 50 epoch |  9.55M |  2.49G | 1268 | 0.3108 | 14.93 | 0.9848 | 39.74 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_tau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_tau_cos_ep50.log) |

### **Benchmark of MetaFormers on SimVP (MetaVP)**

Similar to [Moving MNIST Benchmarks](#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) with training times of 50-epoch. We provide config files in [configs/taxibj/simvp](https://github.com/chengtan9907/OpenSTL/configs/taxibj/simvp/).

| MetaFormer       |  Setting | Params | FLOPs |  FPS |   MSE  |  MAE  |  SSIM  |  PSNR |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:------:|:-----:|:------:|:-----:|:------------:|
| SimVP+IncepU     | 50 epoch | 13.79M | 3.61G |  533 | 0.3282 | 15.45 | 0.9835 | 39.45 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_incepu_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_incepu_cos_ep50.log) |
| SimVP+gSTA-S     | 50 epoch |  9.96M | 2.62G | 1217 | 0.3246 | 15.03 | 0.9844 | 39.71 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_gsta_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_gsta_cos_ep50.log) |
| ViT              | 50 epoch |  9.66M | 2.80G | 1301 | 0.3171 | 15.15 | 0.9841 | 39.64 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_vit_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_vit_cos_ep50.log) |
| Swin Transformer | 50 epoch |  9.66M | 2.56G | 1506 | 0.3128 | 15.07 | 0.9847 | 39.65 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_swin_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_swin_cos_ep50.log) |
| Uniformer        | 50 epoch |  9.52M | 2.71G | 1333 | 0.3268 | 15.16 | 0.9844 | 39.64 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_uniformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_uniformer_cos_ep50.log) |
| MLP-Mixer        | 50 epoch |  8.24M | 2.18G | 1974 | 0.3206 | 15.37 | 0.9841 | 39.49 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_mlpmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_mlpmixer_cos_ep50.log) |
| ConvMixer        | 50 epoch |  0.84M | 0.23G | 4793 | 0.3634 | 15.63 | 0.9831 | 39.41 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_convmixer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_convmixer_cos_ep50.log) |
| Poolformer       | 50 epoch |  7.75M | 2.06G | 1827 | 0.3273 | 15.39 | 0.9840 | 39.46 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_poolformer_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_poolformer_cos_ep50.log) |
| ConvNeXt         | 50 epoch |  7.84M | 2.08G | 1918 | 0.3106 | 14.90 | 0.9845 | 39.76 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_convnext_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_convnext_cos_ep50.log) |
| VAN              | 50 epoch |  9.48M | 2.49G | 1273 | 0.3125 | 14.96 | 0.9848 | 39.72 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_van_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_van_cos_ep50.log) |
| HorNet           | 50 epoch |  9.68M | 2.54G | 1350 | 0.3186 | 15.01 | 0.9843 | 39.66 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_hornet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_hornet_cos_ep50.log) |
| MogaNet          | 50 epoch |  9.96M | 2.61G | 1005 | 0.3114 | 15.06 | 0.9847 | 39.70 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_moganet_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_simvp_moganet_cos_ep50.log) |
| TAU              | 50 epoch |  9.55M | 2.49G | 1268 | 0.3108 | 14.93 | 0.9848 | 39.74 | [model](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_tau_cos_ep50.pth) \| [log](https://github.com/chengtan9907/OpenSTL/releases/download/taxibj-weights/taxibj_tau_cos_ep50.log) |

<p align="right">(<a href="#top">back to top</a>)</p>
