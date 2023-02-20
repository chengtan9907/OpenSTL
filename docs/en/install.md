# Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/chengtan9907/SimVPv2
cd SimVPv2
conda env create -f environment.yml
conda activate SimVP
python setup.py develop
```

<details close>
<summary>Dependencies</summary>

* argparse
* numpy
* hickle
* scikit-image=0.16.2
* torch
* timm
* tqdm
</details>
