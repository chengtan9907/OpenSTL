dataset_parameters = {
    'mmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20
    },
    'taxibj': {
        'in_shape': [4, 2, 32, 32],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8
    },
    **dict.fromkeys(['kth20', 'kth'], {
        'in_shape': [10, 1, 128, 128],
        'pre_seq_length': 10,
        'aft_seq_length': 20,
        'total_length': 30
    }),
    'kth40': {
        'in_shape': [10, 1, 128, 128],
        'pre_seq_length': 10,
        'aft_seq_length': 40,
        'total_length': 50
    },
    'human': {
        'in_shape': [4, 3, 128, 128],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8
    },
    'kitticaltech': {
        'in_shape': [10, 3, 128, 160],
        'pre_seq_length': 10,
        'aft_seq_length': 1,
        'total_length': 11
    },
    **dict.fromkeys(['weather', 'weather_t2m'], {
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 't2m',
        'train_time': ['2010', '2015'],
        'val_time': ['2016', '2016'],
        'test_time': ['2017', '2018']
    })
}