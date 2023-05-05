import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader

try:
    import xarray as xr
except ImportError:
    xr = None

d2r = np.pi / 180


def latlon2xyz(lat, lon):
    if type(lat) == torch.Tensor:
        x = -torch.cos(lat)*torch.cos(lon)
        y = -torch.cos(lat)*torch.sin(lon)
        z = torch.sin(lat)

    if type(lat) == np.ndarray:
        x = -np.cos(lat)*np.cos(lon)
        y = -np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
    return x, y, z


def xyz2latlon(x, y, z):
    if type(x) == torch.Tensor:
        lat = torch.arcsin(z)
        lon = torch.atan2(-y, -x)

    if type(x) == np.ndarray:
        lat = np.arcsin(z)
        lon = np.arctan2(-y, -x)
    return lat, lon


data_map = {'z': 'geopotential_500',
            't': 'temperature_850',
            'tp': 'total_precipitation',
            't2m': '2m_temperature',
            'r': 'relative_humidity',
            'u10': '10m_u_component_of_wind',
            'v10': '10m_v_component_of_wind',
            'tcc': 'total_cloud_cover'}


class WeatherBenchDataset(Dataset):
    """Wheather Bench Dataset <http://arxiv.org/abs/2002.00469>`_

    Args:
        data_root (str): Path to the dataset.
        data_name (str): Name of the weather modality in Wheather Bench.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        data_split (str): The resolution (degree) of Wheather Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, step, data_split='5_625',
                 mean=None, std=None,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.dataname = data_name
        self.data_split = data_split
        self.training_time = training_time
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.mean = mean
        self.std = std
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment

        self.time = None
        shape = int(32 * 5.625 / float(data_split.replace('_', '.')))
        self.shape = (shape, shape * 2)

        if isinstance(data_name, list):
            data_name = data_name[0]

        if data_name != 'uv10':
            try:
                dataset = xr.open_mfdataset(data_root+'/{}/{}*.nc'.format(
                    data_map[data_name], data_map[data_name]), combine='by_coords')
            except (AttributeError, ValueError):
                assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                    'pip install xarray==0.19.0,' \
                                    'pip install netcdf4 h5netcdf dask'
            except OSError:
                print("OSError: Invalid path {}/{}/*.nc".format(data_root, data_map[data_name]))
                assert False
            dataset = dataset.sel(time=slice(*training_time))
            dataset = dataset.isel(time=slice(None, -1, step))
            if self.time is None:
                self.week = dataset['time.week']
                self.month = dataset['time.month']
                self.year = dataset['time.year']
                self.time = np.stack(
                    [self.week, self.month, self.year], axis=1)
                lon, lat = np.meshgrid(
                    (dataset.lon-180) * d2r, dataset.lat*d2r)
                x, y, z = latlon2xyz(lat, lon)
                self.V = np.stack([x, y, z]).reshape(3, self.shape[0]*self.shape[1]).T
                # input_datasets.append(dataset.get(key).values[:, np.newaxis, :, :])
            # self.data = np.concatenate(input_datasets, axis=1)
            self.data = dataset.get(data_name).values[:, np.newaxis, :, :]

        elif data_name == 'uv10':
            input_datasets = []
            for key in ['u10', 'v10']:
                try:
                    dataset = xr.open_mfdataset(data_root+'/{}/{}*.nc'.format(
                        data_map[key], data_map[key]), combine='by_coords')
                except (AttributeError, ValueError):
                    assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                     'pip install xarray==0.19.0,' \
                                     'pip install netcdf4 h5netcdf dask'
                except OSError:
                    print("OSError: Invalid path {}/{}/*.nc".format(data_root, data_map[key]))
                    assert False
                dataset = dataset.sel(time=slice(*training_time))
                dataset = dataset.isel(time=slice(None, -1, step))
                if self.time is None:
                    self.week = dataset['time.week']
                    self.month = dataset['time.month']
                    self.year = dataset['time.year']
                    self.time = np.stack(
                        [self.week, self.month, self.year], axis=1)
                    lon, lat = np.meshgrid(
                        (dataset.lon-180) * d2r, dataset.lat*d2r)
                    x, y, z = latlon2xyz(lat, lon)
                    self.V = np.stack([x, y, z]).reshape(3, self.shape[0]*self.shape[1]).T
                input_datasets.append(dataset.get(key).values[:, np.newaxis, :, :])
            self.data = np.concatenate(input_datasets, axis=1)

        # uv10
        if len(self.data.shape) == 5:
            self.data = self.data.squeeze(1)
        # humidity
        self.data = self.data[:, -1:, ...] if data_name == 'r' else self.data

        if self.mean is None:
            self.mean = self.data.mean(axis=(0, 2, 3)).reshape(
                1, self.data.shape[1], 1, 1)
            self.std = self.data.std(axis=(0, 2, 3)).reshape(
                1, self.data.shape[1], 1, 1)
            # self.mean = dataset.mean('time').mean(('lat', 'lon')).compute()[data_name].values
            # self.std = dataset.std('time').mean(('lat', 'lon')).compute()[data_name].values

        self.data = (self.data-self.mean)/self.std

        self.valid_idx = np.array(
            range(-idx_in[0], self.data.shape[0]-idx_out[-1]-1))

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in])
        labels = torch.tensor(self.data[index+self.idx_out])
        if self.use_augment:
            len_data = self.idx_in.shape[0]
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              data_split='5_625',
              data_name='t2m',
              train_time=['1979', '2015'],
              val_time=['2016', '2016'],
              test_time=['2017', '2018'],
              idx_in=[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
              idx_out=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              step=1,
              distributed=False, use_augment=False, use_prefetcher=False,
              **kwargs):

    assert data_split in ['5_625', '2_8125', '1_40625']
    _dataroot = osp.join(data_root, f'weather_{data_split}deg')
    weather_dataroot = _dataroot if osp.exists(_dataroot) else osp.join(data_root, 'weather')

    train_set = WeatherBenchDataset(data_root=weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=train_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, use_augment=use_augment)
    vali_set = WeatherBenchDataset(weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=val_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)
    test_set = WeatherBenchDataset(weather_dataroot,
                                    data_name, data_split=data_split,
                                    training_time=test_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set, # validation_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    data_split=['5_625', '1_40625']
    data_name = 't2m'

    for _split in data_split:
        dataloader_train, _, dataloader_test = \
            load_data(batch_size=128,
                    val_batch_size=32,
                    data_root='../../data',
                    num_workers=4, data_name=data_name,
                    data_split=_split,
                    train_time=['1979', '2015'],
                    val_time=['2016', '2016'],
                    test_time=['2017', '2018'],
                    idx_in=[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
                    idx_out=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], step=24, use_augment=True)

        print(len(dataloader_train), len(dataloader_test))
        for item in dataloader_train:
            print(item[0].shape)
            break
        for item in dataloader_test:
            print(item[0].shape)
            break
