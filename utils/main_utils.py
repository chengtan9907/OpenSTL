import os
import logging
import numpy as np
import torch
import random 
import torch.backends.cudnn as cudnn
from .config_utils import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


def get_dataset(dataname, config):
    from constants import dataset_parameters
    from api.dataloader import load_data
    config.update(dataset_parameters[dataname])
    return load_data(**config)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(filename:str = None):
    '''
    load and print config
    '''
    print('loading config from ' + filename + ' ...')
    configfile = Config(filename=filename)
    config = configfile._cfg_dict
    return config


def update_config(args, config):
    '''
    update config with a new args dict
    '''
    assert isinstance(args, dict) and isinstance(config, dict)
    for k in config.keys():
        if args.get(k, False):
            if args[k] != config[k]:
                print(f'overwrite config key -- {k}: {config[k]} -> {args[k]}')
        else:
            args[k] = config[k]
    return args
