# Weather Prediction Visualization

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on famous weather prediction datasets, WeatherBench. More STL methods will be supported in the future. Issues and PRs are welcome!**  Visualization of *GIF* is released.

## Table of Contents

- [WeatherBench Benchmarks Visualization](#weatherbench-benchmarks-visualization)
    - [Visualization of STL Benchmarks on Temperature (t2m)](#visualization-of-stl-benchmarks-on-temperature-t2m)
    - [Visualization of STL Benchmarks on Humidity (r)](#visualization-of-stl-benchmarks-on-humidity-r)
    - [Visualization of STL Benchmarks on Wind Component (uv10)](#visualization-of-stl-benchmarks-on-wind-component-uv10)
    - [Visualization of STL Benchmarks on Cloud Cover (tcc)](#visualization-of-stl-benchmarks-on-cloud-cover-tcc)

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

We provide visualization figures of various weather prediction methods on Weather Bench (single variable). You can plot your own visualization with tested results (e.g., `work_dirs/exp_name/saved`) by [vis_video.py](https://github.com/chengtan9907/OpenSTL/tree/master/tools/visualizations/vis_video.py). Note that `--vis_dirs` denotes visualize all experimental folders under the path, and `--vis_channel` can select the channel for visualization. For example, run plotting with the script:
```shell
python tools/visualizations/vis_video.py -d weather_t2m_5_625 -w work_dirs/exp_name --index 0 --save_dirs fig_w_t2m_5_625_vis
```

## WeatherBench Benchmarks Visualization

We provide temperature prediction benchmark results on the popular [WeatherBench](https://arxiv.org/abs/2002.00469) dataset (temperature prediction `t2m`) using $12\rightarrow 12$ frames prediction setting.

### **Visualization of STL Benchmarks on Temperature (t2m)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular STL methods training 50 epochs with **single GPU** on `t2m` (K) in [configs/weather/t2m_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/t2m_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| ConvLSTM |
| :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_ConvLSTM.gif' height="auto" width="260" ></div> |

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>


### **Visualization of STL Benchmarks on Humidity (r)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark STL methods training 50 epochs with **single GPU** on `r` (%). We provide config files in [configs/weather/r_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/r_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| ConvLSTM |
| :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_ConvLSTM.gif' height="auto" width="260" ></div> |

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>


### **Visualization of STL Benchmarks on Wind Component (uv10)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular STL methods training 50 epochs with **single GPU** on `uv10` (ms-1). We provide config files in [configs/weather/uv10_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/uv10_5_625/) for `5.625` settings ($32\times 64$ resolutions). Notice that the input data of `uv10` has two channels.

| ConvLSTM (in flow) | ConvLSTM (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_ConvLSTM.gif' height="auto" width="260" ></div> |

| E3D-LSTM | E3D-LSTM (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_E3DLSTM.gif' height="auto" width="260" ></div> |

| MAU (in flow) | MAU (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_MAU.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_MAU.gif' height="auto" width="260" ></div> |

| MIM (in flow) | MIM (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_MIM.gif' height="auto" width="260" ></div> |

| PhyDNet (in flow) | PhyDNet (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_PhyDNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN (in flow) | PredRNN (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_PredRNN.gif' height="auto" width="260" ></div> |

| PredRNN++ (in flow) | PredRNN++ (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_PredRNNpp.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 (in flow) | PredRNN-V2 (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_PredRNNv2.gif' height="auto" width="260" ></div> |

| SimVP-V1 (in flow) | SimVP-V1 (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_IncepU.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 (in flow) | SimVP-V2 (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_gSTA.gif' height="auto" width="260" ></div> |

| TAU (in flow) | TAU (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_TAU.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer (in flow) | SimVP-ConvMixer (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_Convmixer.gif' height="auto" width="260" ></div> |

| SimVP-ConvNeXt (in flow) | SimVP-ConvNeXt (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_ConvNext.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_ConvNext.gif' height="auto" width="260" ></div> |

| SimVP-HorNet (in flow) | SimVP-HorNet (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_HorNet.gif' height="auto" width="260" ></div> |

| SimVP-MLPMixer (in flow) | SimVP-MLPMixer (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_MLPMixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet (in flow) | SimVP-MogaNet (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_MogaNet.gif' height="auto" width="260" ></div> |

| SimVP-Poolformer (in flow) | SimVP-Poolformer (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_Poolformer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Uniformer (in flow) | SimVP-Uniformer (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_Uniformer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN (in flow) | SimVP-VAN (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_VAN.gif' height="auto" width="260" ></div> |

| SimVP-ViT (in flow) | SimVP-ViT (out flow) |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_SimVP_ViT.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_longtitude_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>


### **Visualization of STL Benchmarks on Cloud Cover (tcc)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular STL methods training 50 epochs with **single GPU** on `tcc` (%). We provide config files in [configs/weather/tcc_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/tcc_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| ConvLSTM |
| :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_ConvLSTM.gif' height="auto" width="260" ></div> |

| E3D-LSTM | MAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_E3DLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_MAU.gif' height="auto" width="260" ></div> |

| MIM | PhyDNet |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_MIM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_PhyDNet.gif' height="auto" width="260" ></div> |

| PredRNN | PredRNN++ |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_PredRNN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_PredRNNpp.gif' height="auto" width="260" ></div> |

| PredRNN-V2 | SimVP-V1 |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_PredRNNv2.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_IncepU.gif' height="auto" width="260" ></div> |

| SimVP-V2 | TAU |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_gSTA.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_TAU.gif' height="auto" width="260" ></div> |

| SimVP-ConvMixer | SimVP-ConvNeXt |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_Convmixer.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_ConvNeXt.gif' height="auto" width="260" ></div> |

| SimVP-HorNet | SimVP-MLPMixer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_HorNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_MLPMixer.gif' height="auto" width="260" ></div> |

| SimVP-MogaNet | SimVP-Poolformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_MogaNet.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_Poolformer.gif' height="auto" width="260" ></div> |

| SimVP-Swin | SimVP-Uniformer |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_Swin.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_Uniformer.gif' height="auto" width="260" ></div> |

| SimVP-VAN | SimVP-ViT |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_VAN.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_SimVP_ViT.gif' height="auto" width="260" ></div> |

<p align="right">(<a href="#top">back to top</a>)</p>
