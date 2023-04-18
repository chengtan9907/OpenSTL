# Copyright (c) CAIRI AI Lab. All rights reserved

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
    args = create_parser().parse_args()
    config = args.__dict__

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    assert args.config_file is not None, "Config file is required for testing"
    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method', 'batch_size', 'val_batch_size'])
    config['test'] = True

    # set multi-process settings
    setup_multi_processes(config)

    print('>'*35 + ' testing  ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()

    mse = exp.test()
    if rank == 0 and has_nni:
        nni.report_final_result(mse)
