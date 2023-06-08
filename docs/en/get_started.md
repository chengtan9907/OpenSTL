# Getting Started

This page provides basic tutorials about the usage of OpenSTL with various spatioTemporal predictive learning (STL) tasks. For installation instructions, please see [Install](docs/en/install.md).

## Training and Testing with a Single GPU

You can perform single GPU training and testing with `tools/train.py` and `tools/test.py` with non-distributed and distributed (DDP) modes. Non-distributed mode is recommanded for the single GPU training (a bit faster than DDP). We provide descriptions of some essential arguments. Other arguments related to datasets, optimizers, methods can be found in [parser.py](https://github.com/chengtan9907/OpenSTL/tree/master/openstl/utils/parser.py).

```bash
python tools/train.py \
    --dataname ${DATASET_NAME} \
    --method ${METHOD_NAME} \
    --config_file ${CONFIG_FILE} \
    --overwrite \
    --ex_name ${EXP_NAME} \
    --resume_from ${CHECKPOINT_FILE} \
    --auto_resume \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --dist \
    --fp16 \
    --seed ${SEED} \
    --clip_grad ${VALUE} \
    --find_unused_parameters \
    --deterministic \
```

**Description of arguments**:
- `--dataname (-d)` : The name of dataset, default to be `mmnist`.
- `--method (-m)` : The name of the video prediction method to train or test, default to be `SimVP`.
- `--config_file (-c)` : The path of a model config file, which will provide detailed settings for a STL method.
- `--overwrite` : Whether to overwrite predefined args in the config file.
- `--ex_name` : The name of the experiment under the `res_dir`. Default to be `Debug`.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file. Or you can use `--auto_resume` to resume from `latest.pth` automatically.
- `--auto_resume` : Whether to automatically resume training when the experiment was interrupted.
- `--batch_size (-b)` : Training batch size, default to 16.
- `--lr` : The basic training learning rate, defaults to 0.001.
- `--dist`: Whether to use distributed training (DDP).
- `--fp16`: Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0).
- `--seed ${SEED}`: Setup all random seeds to a certain number (defaults to 42).
- `--clip_grad ${VALUE}`: Clip gradient norm value (default: None, no clipping).
- `--find_unused_parameters`: Whether to find unused parameters in forward during DDP training.
- `--deterministic`: Switch on "deterministic" mode, which slows down training while the results are reproducible.

An example of single GPU (non-distributed) training with SimVP+gSTA on Moving MNIST dataset.
```shell
bash tools/prepare_data/download_mmnist.sh
python tools/train.py -d mmnist --lr 1e-3 -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

An example of single GPU testing with SimVP+gSTA on Moving MNIST dataset.
```shell
python tools/test.py -d mmnist -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

## Training and Testing with Multiple GPUs

For larger STL tasks (e.g., high resolutions), you can also perform multiple GPUs training and testing with `tools/dist_train.sh` and `tools/dist_test.sh` with DDP mode. The bash files will call `tools/train.py` and `tools/test.py` with the necessary arguments.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
**Description of arguments**:
- `${CONFIG_FILE}` : The path of a model config file, which will provide detailed settings for a STL method.
- `${GPUS}` : The number of GPUs for DDP training.

Examples of multiple GPUs training on Moving MNIST dataset with a machine with 8 GPUs. Note that some recurrent-based STL methods (e.g., ConvLSTM, PredRNN++) need `--find_unused_parameters` during DDP training.
```shell
PORT=29001 CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/mmnist/simvp/SimVP_gSTA.py 2 -d mmnist --lr 1e-3 --batch_size 8
PORT=29002 CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train.sh configs/mmnist/ConvLSTM.py 2 -d mmnist --lr 1e-3 --batch_size 8 --find_unused_parameters
PORT=29003 CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh configs/mmnist/PredRNNpp.py 4 -d mmnist --lr 1e-3 --batch_size 4 --find_unused_parameters
```

An example of multiple GPUs testing on Moving MNIST dataset. The bash script is `bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} ${CHECKPOINT} [optional arguments]`, where the first three augments are necessary.
```shell
PORT=29001 CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_test.sh configs/mmnist/simvp/SimVP_gSTA.py 2 work_dirs/mmnist/simvp/SimVP_gSTA -d mmnist
```

**Note**:
* During DDP training, the number of GPUS `ngpus` should be provided, and checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/` (it will be the default setting if `--ex_name` is not specified). The default learning rate `lr` and the batch size `bs` in config files are for a single GPU. If using a different number GPUs, the total batch size will change in proportion, you have to scale the learning rate following `lr = base_lr * ngpus` and `bs = base_bs * ngpus` (known as the `linear scaling rule`). Other arguments should be added as the single GPU training.
* Experiment results using different GPUs settings will produce different results. We have noticed that single GPU training with DP and DDP setups will produce similar results, while different multiple GPUs using the **linear scaling rule** will cause different results because of DDP training. For example, SimVP+gSTA is trained 200 epochs on MMNIST with `1GPU (DP)`, `1GPU (DDP)`, `2GPUs (2xbs8)`, and `4GPUs (4xbs4)` using the same learning rate (lr=1e-3), we produce results of MSE 26.73, 26.78, 30.01, 31.36. Therefore, we will provide the used GPUs setting in the benchmark result with the corresponding learning rate for fair comparison and reproducible purposes.
* DDP training and testing error of `WARNING:torch.distributed.elastic.multiprocessing.api:Sending process xxx closing signal SIGTERM` might sometimes occur, caused by PyTorch1.9 to PyTorch1.10. You can use PyTorch1.8 or PyTorch2.0 to get rid of these errors, or conduct `1GPU` experiments.

## Mixed Precision Training

We support Mixed Precision Training implemented by PyTorch AMP. If you want to use Mixed Precision Training, you can add `--fp16` in the arguments.
