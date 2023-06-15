## Changelog

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
* Fix the bug of `PSNR` (changing the implementation from E3D-LSTM to the corrent version) and update results in the benchmarks.

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

* Upload `readthedocs` documents. Summarize video prediction benchmark results on MMNIST in [video_benchmarks.md](https://github.com/chengtan9907/SimVPv2/docs/en/model_zoos/video_benchmarks.md).
* Update benchmark results of video prediction baselines and MetaFormer architectures based on SimVP on MMNIST, TaxiBJ, and WeatherBench datasets.
* Update README and add a license.
