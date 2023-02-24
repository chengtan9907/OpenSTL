# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from simvp.api import NonDistExperiment
from simvp.utils import create_parser, load_config, update_config

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
                           exclude_keys=['batch_size', 'val_batch_size'])
    config['test'] = True

    exp = NonDistExperiment(args)

    print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()
    if has_nni:
        nni.report_final_result(mse)
