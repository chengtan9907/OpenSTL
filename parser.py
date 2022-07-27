import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--val_batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist',
                        choices=['mmnist'])

    # method parameters
    parser.add_argument('--method', default='SimVP', choices=[
                        'SimVP', 'ConvLSTM', 'PredRNNpp', 'PredRNN', 'PredRNNv2', 'MIM', 'E3DLSTM', 'MAU', 'CrevNet', 'PhyDNet'])
    parser.add_argument('--config_file', '-c', default='./configs/SimVP.py', type=str)

    # Training parameters
    parser.add_argument('--epoch', default=201, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    return parser