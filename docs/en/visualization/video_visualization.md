# Video Prediction Visualization

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on popular traffic prediction datasets. More STL methods will be supported in the future. Issues and PRs are welcome!** Visualization of *GIF* will be released soon (you can contact us if you require these files).

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


## Moving MNIST Benchmarks

We provide benchmark results on the popular [Moving MNIST](http://arxiv.org/abs/1502.04681) dataset using $10\rightarrow 10$ frames prediction setting following [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855). Metrics (MSE, MAE, SSIM, pSNR) of the the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Onecycle scheduler and **single GPU**.

## Moving FMNIST Benchmarks

Similar to [Moving MNIST](http://arxiv.org/abs/1502.04681), we also provide the advanced version of MNIST, i.e., MFMNIST benchmark results, using $10\rightarrow 10$ frames prediction setting following [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855). Metrics (MSE, MAE, SSIM, pSNR) of the the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Onecycle scheduler and **single GPU**.

## KittiCaltech Benchmarks

We provide benchmark results on [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297) dataset using $10\rightarrow 1$ frames prediction setting following [PredNet](https://arxiv.org/abs/1605.08104). Metrics (MSE, MAE, SSIM, pSNR, LPIPS) of the the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. The default training setup is trained 100 epochs by Adam optimizer with Onecycle scheduler on **single GPU**, while some computational consuming methods (denoted by \*) using **4GPUs**.

<p align="right">(<a href="#top">back to top</a>)</p>
