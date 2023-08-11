# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':
    parser = create_parser()
    
    #Arguments priority: 
    #   command line > f{args.method}.py > default from parser.py
    #first parser run to get the parameters to load f{args.method}.py 
    args = parser.parse_args()
    config = args.__dict__

    #If we provided a config file, loads it. Else, tries to find one for
    #the method.
    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    loaded_cfg = load_config(cfg_path)
    #Push the default values for this method to the parser
    parser.set_defaults(**loaded_cfg)
    
    #second parser run to get the final parameters
    args = parser.parse_args()

    # set multi-process settings
    setup_multi_processes(config)

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()

    if rank == 0 and has_nni:
        nni.report_final_result(mse)
