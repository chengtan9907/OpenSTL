# Getting Started

This page provides basic tutorials about the usage of SimVP. For installation instructions, please see [Install](docs/en/install.md).

## Training and Testing with a Single GPU

You can perform single GPU training and testing with `tools/non_dist_train.py` and `tools/non_dist_test.py`. We provide descriptions of some essential arguments.

```bash
python tools/non_dist_train.py \
    --dataname ${DATASET_NAME} \
    --method ${METHOD_NAME} \
    --config_file ${CONFIG_FILE} \
    --ex_name ${EXP_NAME} \
    --auto_resume \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
```

**Description of arguments**:
- `--dataname (-d)` : The name of dataset, default to be `mmnist`.
- `--method (-m)` : The name of the video prediction method to train or test, default to be `SimVP`.
- `--config_file (-c)` : The path of a model config file, which will provide detailed settings for a video prediction method.
- `--ex_name` : The name of the experiment under the `res_dir`. Default to be `Debug`.
- `--auto_resume` : Whether to automatically resume training when the experiment was interrupted.
- `--batch_size (-b)` : Training batch size, default to 16.
- `--lr` : The basic training learning rate, defaults to 0.001.

An example of single GPU training with SimVP+gSTA on Moving MNIST dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python tools/non_dist_train.py -d mmnist --lr 1e-3 -c ./configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

An example of single GPU testing with SimVP+gSTA on Moving MNIST dataset.
```shell
python tools/non_dist_test.py -d mmnist -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```
