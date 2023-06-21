# Weather Prediction Visualization

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on famous weather prediction datasets, WeatherBench. More STL methods will be supported in the future. Issues and PRs are welcome!**  Visualization of *GIF* is released.

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

We provide visualization figures of various weather prediction methods on Weather Bench (single variable). You can plot your own visualization with tested results (e.g., `work_dirs/EXP/saved`) by [vis_video.py](https://github.com/chengtan9907/OpenSTL/tree/master/tools/visualizations/vis_video.py). For example, run plotting with the script:
```shell
python tools/visualizations/vis_video.py -d weather_t2m_5_625 -w work_dirs/EXP/ --index 0 --save_dirs fig_w_t2m_5_625_vis
```

## WeatherBench Benchmarks Visualization

We provide temperature prediction benchmark results on the popular [WeatherBench](https://arxiv.org/abs/2002.00469) dataset (temperature prediction `t2m`) using $12\rightarrow 12$ frames prediction setting.

### **Visualization of STL Benchmarks on Temperature (t2m)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular STL methods training 50 epochs with **single GPU** on `t2m` (K) in [configs/weather/t2m_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/t2m_5_625/) for `5.625` settings ($32\times 64$ resolutions).

<p align="right">(<a href="#top">back to top</a>)</p>


### **Visualization of STL Benchmarks on Humidity (r)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark STL methods training 50 epochs with **single GPU** on `r` (%). We provide config files in [configs/weather/r_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/r_5_625/) for `5.625` settings ($32\times 64$ resolutions).

<p align="right">(<a href="#top">back to top</a>)</p>


### **Visualization of STL Benchmarks on Wind Component (uv10)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular STL methods training 50 epochs with **single GPU** on `uv10` (ms-1). We provide config files in [configs/weather/uv10_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/uv10_5_625/) for `5.625` settings ($32\times 64$ resolutions). Notice that the input data of `uv10` has two channels.

<p align="right">(<a href="#top">back to top</a>)</p>


### **Visualization of STL Benchmarks on Cloud Cover (tcc)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular STL methods training 50 epochs with **single GPU** on `tcc` (%). We provide config files in [configs/weather/tcc_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/tcc_5_625/) for `5.625` settings ($32\times 64$ resolutions).

<p align="right">(<a href="#top">back to top</a>)</p>
