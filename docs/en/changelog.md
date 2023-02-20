## Changelog

### v0.1.0 (18/02/2023)

Release version to V0.1.0 with code refactoring.

#### Code Refactoring

* Refactor code structures as `simvp/api`, `simvp/core`, `simvp/datasets`, `simvp/methods`, `simvp/models`, `simvp/modules`. We support non-distributed training and evaluation by the executable python file `tools/non_dist_train.py`. Refactor config files for SimVP models.
* Fix bugs in tools/nondist_train.py, simvp/utils, environment.yml, and .gitignore, etc.

#### New Features

* Support Timm optimizers and schedulers.
* Update popular Metaformer models as the hidden Translator $h$ in SimVP, supporting [ViT](https://arxiv.org/abs/2010.11929), [Swin-Transformer](https://arxiv.org/abs/2103.14030), [MLP-Mixer](https://arxiv.org/abs/2105.01601), [ConvMixer](https://arxiv.org/abs/2201.09792), [UniFormer](https://arxiv.org/abs/2201.09450), [PoolFormer](https://arxiv.org/abs/2111.11418), [ConvNeXt](https://arxiv.org/abs/2201.03545), [VAN](https://arxiv.org/abs/2202.09741), [HorNet](https://arxiv.org/abs/2207.14284), and [MogaNet](https://arxiv.org/abs/2211.03295).
* Update implementations of dataset and dataloader, supporting [KTH Action](https://ieeexplore.ieee.org/document/1334462), [KittiCaltech Pedestrian](https://dl.acm.org/doi/10.1177/0278364913491297), [Moving MNIST](http://arxiv.org/abs/1502.04681), [TaxiBJ](https://arxiv.org/abs/1610.00081), and [WeatherBench](https://arxiv.org/abs/2002.00469).

### Update Documents

* Upload `readthedocs` documents. Summarize video prediction benchmark results on MMNIST in [video_benchmarks.md](https://github.com/chengtan9907/SimVPv2/docs/en/model_zoos/video_benchmarks.md).
