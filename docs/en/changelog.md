## Changelog

### v1.0.0 (15/12/2023)

Release version to OpenSTL (PyTorch Lightning) V1.0.0.

#### New Features

* Update the training and testing pipelines for OpenSTL based on PyTorch Lightning.
* Support more STL methods, e.g., WaST.

#### Update Documents

* Fix the readthedoc version of the [webpage](https://openstl.readthedocs.io/en/latest/).

### v0.3.0 (19/06/2023)

Release version to OpenSTL V0.3.0 as [#25](https://github.com/chengtan9907/OpenSTL/issues/25).

#### New Features

* Support visualization tools in [vis_video](https://github.com/chengtan9907/OpenSTL/tree/master/tools/visualizations/vis_video.py), config files in [configs](https://github.com/chengtan9907/OpenSTL/tree/master/configs), and trained files (models, logs, and visualizations) in [v0.3.0](https://github.com/chengtan9907/OpenSTL/releases/tag/v0.3.0) of STL methods on various datasets (on updating).
* Support the dataloader of video classification datasets [Kinetics](https://deepmind.com/research/open-source/kinetics) and [BAIR](https://arxiv.org/abs/1710.05268), which has a similar setting as the Human3.6M and KTH dataloaders. Relevant video transforms in Kinetics are supported according to [VideoMAE](https://github.com/MCG-NJU/VideoMAE), and config files are provided. Add data preparation of TaxiBJ as issue [#34](https://github.com/chengtan9907/OpenSTL/issues/34).
* Update STL results visualization by [vis_video](https://github.com/chengtan9907/OpenSTL/tree/master/tools/visualizations/vis_video.py) for video prediction, traffic prediction, weather prediction tasks in [video_visualization](https://github.com/chengtan9907/OpenSTL/docs/en/visualization/video_visualization.md), [traffic_visualization](https://github.com/chengtan9907/OpenSTL/docs/en/visualization/traffic_visualization.md), and [weather_visualization](https://github.com/chengtan9907/SimVPv2/docs/en/visualization/weather_visualization.md).
* Support Jupyter notebook tutorials and video examples in [examples](https://github.com/chengtan9907/OpenSTL/tree/master/examples).
* Support early-stop training with `--early_stop_epoch` as issue [#36](https://github.com/chengtan9907/OpenSTL/issues/36).
* Support inference only with `--inference` in `tools/test.py` for issue [#55](https://github.com/chengtan9907/OpenSTL/issues/55), where results will be saved in `ex_name/saved`.

#### Update Documents

* The [OpenSTL](https://arxiv.org/abs/2306.11249) paper has been accepted by NeurIPS 2023 Dataset and Benchmark Track.
* Release arXiv preprint of [OpenSTL](https://arxiv.org/abs/2306.11249), which describes the overall framework, benchmark results, and experimental settings, etc.
* Update benchmark results of video prediction, traffic prediction, and weather prediction benchmarks in `docs/en/model_zoos`.
* Add the Huggingface organization for [OpenSTL🤗](https://huggingface.co/OpenSTL), where users can join it by [invitation link](https://huggingface.co/organizations/OpenSTL/share/ovCzbEGVhnQNFHBGMMLfXEsPhmuqgBZfii).

#### Fix Bugs

* Fix bugs in the dataloader (issue [#26](https://github.com/chengtan9907/OpenSTL/issues/26)) and dataset prepration tools (issue [#27](https://github.com/chengtan9907/OpenSTL/issues/27) and [#28](https://github.com/chengtan9907/OpenSTL/issues/28)).
* Fix bugs of overwrite config values during training, where `utils/main_utils/update_config` will overwrite the config file with the default values in `utils/main_utils/parser` in mistake (issue [#42](https://github.com/chengtan9907/OpenSTL/issues/42)). Using `default_parser()` to provide the default values and fulfill the config after updating values in the given config file (solving pull request [#47](https://github.com/chengtan9907/OpenSTL/pull/47)).
* Fix bugs of env installation (issue [#62](https://github.com/chengtan9907/OpenSTL/issues/62)) and update `environment.yml`.

### v0.2.0 (21/04/2023)

Release version to OpenSTL V0.2.0 as [#20](https://github.com/chengtan9907/OpenSTL/issues/20).

#### Code Refactoring

* Rename the project to `OpenSTL` instead of `SimVPv2` with module name refactoring.
* Refactor the code structure thoroughly to support non-distributed and distributed (DDP) training & testing with `tools/train.py` and `tools/test.py`.
* Refactor `_dist_forward_collect` and `_non_dist_forward_collect` to support collection of metrics.

#### New Features

* Update the Weather Bench dataloader with `5.625deg`, `2.8125deg`, and `1.40625deg` settings. Add Human3.6M dataloader (supporting augmentations) and config files. Add Moving FMNIST and MMNIST_CIFAR as two advanced variants of MMNIST datasets.
* Update tools for dataset preparation of Human3.6M, Weather Bench, and Moving FMNIST.
* Support [PredNet](https://openreview.net/forum?id=B1ewdt9xe), [TAU](https://arxiv.org/abs/2206.12126), and [DMVFN](https://arxiv.org/abs/2303.09875) with configs and benchmark results. And fix bugs in these new STL methods.
* Support multi-variant versions of Weather Bench with dataloader and metrics.
* Support [lpips](https://github.com/richzhang/PerceptualSimilarity/tree/master) metric for video prediction benchmarks.
* Support STL results visualization by [vis_video](https://github.com/chengtan9907/OpenSTL/tree/master/tools/visualizations/vis_video.py) for video prediction, traffic prediction, weather prediction tasks.
* Support visualization of STL methods on various datasets (on updating).

#### Update Documents

* Update documents of video prediction, traffic prediction, and weather prediction benchmarks with benchmark results and spesific GPU settings (e.g., **single GPU**). Provide config files for supported STL methods.
* Update `docs/en` documents for the basic usages and new features of V0.2.0. Adding detailed steps of installation and preparation datasets.
* Clean-up STL benchmarks and update to the latest results with config files provided.

#### Fix Bugs

* Fix bugs in training loops and validation loops to save GPU memory.
* There might be some bugs in not using all parameters for calculating losses in ConvLSTM CrevNet, which should use `--find_unused_parameters` for DDP training.
* Fig bugs of building distributed dataloaders and preparation of DDP training.
* Fix bugs of some STL methods (CrevNet, DMVFN, PreDNet, and TAU).
* Fix bugs in datasets: fixing Caltech dataset for evaluation (28/05/2023 updating [Baidu Cloud](https://pan.baidu.com/s/1fudsBHyrf3nbt-7d42YWWg?pwd=kjfk)).
* Fix the bug of `PSNR` (changing the implementation from E3D-LSTM to the current version) and update results in the benchmarks.

### v0.1.0 (18/02/2023)

Release version to V0.1.0 with code refactoring.

#### Code Refactoring

* Refactor code structures as `simvp/api`, `simvp/core`, `simvp/datasets`, `simvp/methods`, `simvp/models`, `simvp/modules`. We support non-distributed training and evaluation by the executable python file `tools/non_dist_train.py`. Refactor config files for SimVP models.
* Fix bugs in tools/nondist_train.py, simvp/utils, environment.yml, and .gitignore, etc.

#### New Features

* Support Timm optimizers and schedulers.
* Update popular Metaformer models as the hidden Translator $h$ in SimVP, supporting [ViT](https://arxiv.org/abs/2010.11929), [Swin-Transformer](https://arxiv.org/abs/2103.14030), [MLP-Mixer](https://arxiv.org/abs/2105.01601), [ConvMixer](https://arxiv.org/abs/2201.09792), [UniFormer](https://arxiv.org/abs/2201.09450), [PoolFormer](https://arxiv.org/abs/2111.11418), [ConvNeXt](https://arxiv.org/abs/2201.03545), [VAN](https://arxiv.org/abs/2202.09741), [HorNet](https://arxiv.org/abs/2207.14284), and [MogaNet](https://arxiv.org/abs/2211.03295).
* Update implementations of dataset and dataloader, supporting [KTH Action](https://ieeexplore.ieee.org/document/1334462), [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297), [Moving MNIST](http://arxiv.org/abs/1502.04681), [TaxiBJ](https://arxiv.org/abs/1610.00081), and [WeatherBench](https://arxiv.org/abs/2002.00469).

#### Update Documents

* Release arXiv preprint of [SimVPv2](https://arxiv.org/abs/2211.12509). This version supports the morst experiments in [SimVPv2](https://arxiv.org/abs/2211.12509), which is the extend version of [SimVP](https://arxiv.org/abs/2206.05099).
* Upload `readthedocs` documents. Summarize video prediction benchmark results on MMNIST in [video_benchmarks.md](https://github.com/chengtan9907/SimVPv2/docs/en/model_zoos/video_benchmarks.md).
* Update benchmark results of video prediction baselines and MetaFormer architectures based on SimVP on MMNIST, TaxiBJ, and WeatherBench datasets.
* Update README and add a license.
