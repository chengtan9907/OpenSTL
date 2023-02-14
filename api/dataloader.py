

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    pre_seq_length = kwargs.get('', 10)
    aft_seq_length = kwargs.get('', 10)
    if dataname == 'mmnist':
        from .dataloader_moving_mnist import load_data
        return load_data(batch_size, val_batch_size, num_workers, data_root, pre_seq_length, aft_seq_length)
    elif dataname == 'kitticaltech':
        from .dataloader_kitticaltech import load_data
        return load_data(batch_size, val_batch_size, num_workers, data_root, pre_seq_length, aft_seq_length)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
