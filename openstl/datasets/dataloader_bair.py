import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    import tensorflow as tf
except ImportError:
    tf = None

from openstl.datasets.utils import create_loader


class BAIRDataset(Dataset):
    """ BAIR Robot Pushing Action Dataset
        <https://arxiv.org/abs/1710.05268>`_

    Args:
        datas, indices (list): Data and indices of path.
        image_size (int: The target resolution of Human3.6M images.
        pre_seq_length (int): The input sequence length.
        aft_seq_length (int): The output sequence length for prediction.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, datas, indices, image_size=64, pre_seq_length=2, aft_seq_length=12, use_augment=False, data_name='bair'):
        super(BAIRDataset,self).__init__()
        self.datas = datas
        self.indices = indices
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.tot_seq_length = self.pre_seq_length + self.aft_seq_length
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, imgs, crop_scale=0.95):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [10, 3, 64, 64]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            imgs = torch.flip(imgs, dims=(3, ))  # horizontal flip
        return imgs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind

        input_batch = np.zeros(
            (self.pre_seq_length + self.aft_seq_length, self.image_size, self.image_size, 3)).astype(np.float32)
        begin = batch_ind[-1]
        end = begin + self.tot_seq_length
        k = 0
        for serialized_example in tf.compat.v1.python_io.tf_record_iterator(batch_ind[0]):
            if k == batch_ind[1]:
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                break
            k += 1
        for j in range(begin, end):
            aux1_image_name = str(j) + '/image_aux1/encoded'
            aux1_byte_str = example.features.feature[aux1_image_name].bytes_list.value[0]
            aux1_img = Image.frombytes('RGB', (self.image_size, self.image_size), aux1_byte_str)
            aux1_arr = np.array(aux1_img.getdata()).reshape((aux1_img.size[1], aux1_img.size[0], 3))

            input_batch[j - begin, :, :, :3] = aux1_arr.reshape(self.image_size, self.image_size, 3) / 255

        input_batch = torch.tensor(input_batch).float().permute(0, 3, 1, 2)
        data = input_batch[:self.pre_seq_length, ::]
        labels = input_batch[self.pre_seq_length:self.tot_seq_length, ::]
        if self.use_augment:
            imgs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.95)
            data = imgs[:self.pre_seq_length, ...]
            labels = imgs[self.pre_seq_length:self.tot_seq_length, ...]
        return data, labels


class InputHandle(object):
    """Class for handling dataset inputs."""

    def __init__(self, datas, indices, configs):
        self.name = configs['name']
        self.minibatch_size = configs['minibatch_size']
        self.image_height = configs['image_height']
        self.image_width = configs['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = configs['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        else:
            self.current_batch_indices = self.indices[
                self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        return self.current_position + self.minibatch_size >= self.total()

    def get_batch(self):
        if self.no_batch_left():
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_height, self.image_width, 7)).astype(np.float32)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind[-1]
            end = begin + self.current_input_length
            k = 0
            for serialized_example in tf.compat.v1.python_io.tf_record_iterator(batch_ind[0]):
                if k == batch_ind[1]:
                    example = tf.train.Example()
                    example.ParseFromString(serialized_example)
                    break
                k += 1
            for j in range(begin, end):
                action_name = str(j) + '/action'
                action_value = np.array(example.features.feature[action_name].float_list.value)
                if action_value.shape == (0,):  # End of frames/data
                    print("error! " + str(batch_ind))
                input_batch[i, j - begin, :, :, 3:] = np.stack([np.ones([64, 64]) * i for i in action_value], axis=2)

                aux1_image_name = str(j) + '/image_aux1/encoded'
                aux1_byte_str = example.features.feature[aux1_image_name].bytes_list.value[0]
                aux1_img = Image.frombytes('RGB', (64, 64), aux1_byte_str)
                aux1_arr = np.array(aux1_img.getdata()).reshape((aux1_img.size[1], aux1_img.size[0], 3))

                input_batch[i, j - begin, :, :, :3] = aux1_arr.reshape(64, 64, 3) / 255
        input_batch = input_batch.astype(np.float32)
        return input_batch


class DataProcess(object):
    """Class for preprocessing dataset inputs."""

    def __init__(self, configs):
        self.configs = configs
        self.data_path = configs['data_path']
        self.image_height = configs['image_height']
        self.image_width = configs['image_width']
        self.seq_len = configs['seq_length']

    def load_data(self, path, mode='train'):
        """Loads the dataset.
        Args:
            path: action_path.
            mode: Training or testing.
        Returns:
            A dataset and indices of the sequence.
        """
        assert mode in ['train', 'test']
        if mode == 'train':
            path = os.path.join(path, 'train')
        else:
            path = os.path.join(path, 'test')
        print('begin load data' + str(path))

        video_fullpaths = []
        indices = []

        tfrecords = os.listdir(path)
        tfrecords.sort()
        num_pictures = 0
        assert tf is not None and 'Please install tensorflow, e.g., pip install tensorflow'

        for tfrecord in tfrecords:
            filepath = os.path.join(path, tfrecord)
            video_fullpaths.append(filepath)
            k = 0
            for serialized_example in tf.compat.v1.python_io.tf_record_iterator(os.path.join(path, tfrecord)):
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                i = 0
                while True:
                    action_name = str(i) + '/action'
                    action_value = np.array(example.features.feature[action_name].float_list.value)
                    if action_value.shape == (0,):  # End of frames/data
                        break
                    i += 1
                num_pictures += i
                for j in range(i - self.seq_len + 1):
                    indices.append((filepath, k, j))
                k += 1
        print("there are " + str(num_pictures) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return video_fullpaths, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.data_path, mode='train')
        return InputHandle(train_data, train_indices, self.configs)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.data_path, mode='test')
        return InputHandle(test_data, test_indices, self.configs)


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=2, aft_seq_length=12, in_shape=[2, 3, 64, 64],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    img_height = in_shape[-2] if in_shape is not None else 64
    img_width = in_shape[-1] if in_shape is not None else 64
    input_param = {
        'data_path': os.path.join(data_root, 'softmotion30_44k'),
        'image_height': img_height,
        'image_width': img_width,
        'minibatch_size': batch_size,
        'seq_length': (pre_seq_length + aft_seq_length),
        'input_data_type': 'float32',
        'name': 'bair'
    }
    input_handle = DataProcess(input_param)
    train_input_handle = input_handle.get_train_input_handle()
    test_input_handle = input_handle.get_test_input_handle()

    train_set = BAIRDataset(train_input_handle.datas,
                            train_input_handle.indices,
                            img_height,
                            pre_seq_length, aft_seq_length, use_augment=use_augment)
    test_set = BAIRDataset(test_input_handle.datas,
                           test_input_handle.indices,
                           img_height,
                           pre_seq_length, aft_seq_length, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = None
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=12)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
