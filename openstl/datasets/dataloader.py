# Copyright (c) CAIRI AI Lab. All rights reserved

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_prefetcher=kwargs.get('use_prefetcher', False),
    )
    print('load_data', dataname, cfg_dataloader)
    if dataname == 'human':
        from .dataloader_human import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'kitticaltech':
        from .dataloader_kitticaltech import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'kth' in dataname:  # 'kth', 'kth20', 'kth40'
        from .dataloader_kth import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'mmnist':
        from .dataloader_moving_mnist import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'taxibj':
        from .dataloader_taxibj import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'weather' in dataname:  # 'weather', 'weather_t2m', etc.
        from .dataloader_weather import load_data
        data_split_pool = ['5_625', '2_8125', '1_40625']
        data_split = '5_625'
        for k in data_split_pool:
            if dataname.find(k) != -1:
                data_split = k
        return load_data(batch_size, val_batch_size, data_root, num_workers,
                         distributed=dist, data_split=data_split, **kwargs)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
