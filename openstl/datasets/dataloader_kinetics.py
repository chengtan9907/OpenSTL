import os
import numpy as np
import warnings

from torch.utils.data import Dataset
from openstl.datasets.pipelines.transforms import (Compose, CenterCrop, ClipToTensor,
                                                   Resize, Normalize)
from openstl.datasets.utils import create_loader

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader, cpu = None, None


def int_to_str(num, str_len=6):
    assert isinstance(num, (int, str))
    num = str(num)
    str_num = (str_len - len(num)) * '0' + num
    return str_num


class KineticsDataset(Dataset):
    """ Video Classification Kinetics Dataset
        <https://arxiv.org/abs/1705.06950>`_

    Args:
        data_root (str): Path to the dataset.
        list_path (str): Path to the txt list file.
        image_size (int: The target resolution of Human3.6M images.
        pre_seq_length (int): The input sequence length.
        aft_seq_length (int): The output sequence length for prediction.
        frame_sample_rate (int): Sampling step in the time dimension (defaults to 2).
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, list_path, image_size=256,
                 pre_seq_length=4, aft_seq_length=4, frame_sample_rate=2,
                 keep_aspect_ratio=False, num_segment=1, use_augment=False, data_name='kinetics'):
        super(KineticsDataset,self).__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length = pre_seq_length + aft_seq_length
        self.use_augment = use_augment
        self.data_name = data_name
        self.input_shape = (self.seq_length, self.image_size, self.image_size, 3)

        self.frame_sample_rate = frame_sample_rate
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        assert list_path.endswith('csv')

        import pandas as pd
        file_list = pd.read_csv(list_path, header=None, delimiter=',')
        dataset_samples = list(file_list.values[1:, 1])
        self.label_array = list(file_list.values[1:, 0])
        self.mode = list(file_list.values[1:, 4])[0]
        time_start, time_end = list(file_list.values[1:, 2]), list(file_list.values[1:, 3])
        self.file_list = list()
        for i,name in enumerate(dataset_samples):
            self.file_list.append(os.path.join(data_root, self.mode, "{}_{}_{}.mp4".format(
                name, int_to_str(time_start[i]), int_to_str(time_end[i]))))

        self.data_transform = Compose([
            Resize(image_size, interpolation='bilinear'),
            CenterCrop(size=(image_size, image_size)),
            ClipToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mean = None
        self.std = None

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.image_size, height=self.image_size,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.seq_length:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.seq_length * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(
                    self.seq_length - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.seq_length)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def _augment_seq(self, buffer):
        """Augmentations for video"""
        raise NotImplementedError

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        buffer = self.loadvideo_decord(sample)
        if len(buffer) == 0:
            while len(buffer) == 0:
                warnings.warn("video {} not correctly loaded during {}ing".format(sample, self.mode))
                index = np.random.randint(self.__len__())
                sample = self.file_list[index]
                buffer = self.loadvideo_decord(sample)

        # augmentation
        if self.use_augment:
            buffer = self._augment_seq(buffer)

        # transform
        buffer = self.data_transform(buffer).permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
        data = buffer[0:self.pre_seq_length, ...]
        labels = buffer[self.aft_seq_length:self.seq_length, ...]
        # print(sample, buffer.shape, data.shape, labels.shape)

        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=4, data_name='kinetics400',
              pre_seq_length=4, aft_seq_length=4, in_shape=[4, 3, 256, 256],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    assert data_name in ['kinetics400', 'kinetics600', 'kinetics700']
    data_root = os.path.join(data_root, data_name)
    image_size = in_shape[-1] if in_shape is not None else 256
    train_set = KineticsDataset(data_root, os.path.join(data_root, 'annotations/train.csv'), image_size,
                            pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                            frame_sample_rate=2, keep_aspect_ratio=True, use_augment=use_augment)
    val_set = KineticsDataset(data_root, os.path.join(data_root, 'annotations/val.csv'), image_size,
                            pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                            frame_sample_rate=2, keep_aspect_ratio=True, use_augment=False)
    test_set = KineticsDataset(data_root, os.path.join(data_root, 'annotations/test.csv'), image_size,
                            pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                            frame_sample_rate=2, keep_aspect_ratio=True, use_augment=False)
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_val = create_loader(val_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_val, dataloader_test


if __name__ == '__main__':

    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4,
                  use_prefetcher=True, distributed=True)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
