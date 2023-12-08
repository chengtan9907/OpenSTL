# Copyright (c) CAIRI AI Lab. All rights reserved

import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    assert args.config_file is not None, "Config file is required for testing"
    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method', 'val_batch_size'])
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    print('>'*35 + ' testing  ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()

    exp.test()