

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    pre_seq_length = kwargs.get('pre_seq_length', 10)
    aft_seq_length = kwargs.get('aft_seq_length', 10)
    if dataname == 'kitticaltech':
        from .dataloader_kitticaltech import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length)
    elif dataname in ['kth', 'kth20', 'kth40']:
        from .dataloader_kth import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length)
    elif dataname == 'mmnist':
        from .dataloader_moving_mnist import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length)
    elif dataname == 'taxibj':
        from .dataloader_taxibj import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length)
    elif dataname == 'weather':
        from .dataloader_weather import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
