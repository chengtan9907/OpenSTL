import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class HumanDataset(Dataset):
    """Human 3.6M Dataset

    Args:
        data_root (str): Path to the dataset.
        list_path (str): Path to the txt list file.
        image_size (int: The target resolution of Human3.6M images.
        pre_seq_length (int): The input sequence length.
        aft_seq_length (int): The output sequence length for prediction.
        step (int): Sampling step in the time dimension (defaults to 5).
    """

    def __init__(self, data_root, list_path, image_size=256,
                 pre_seq_length=4, aft_seq_length=4, step=5):
        super(HumanDataset,self).__init__()
        self.data_root = data_root
        self.file_list = None
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length = pre_seq_length + aft_seq_length
        self.step = step
        self.input_shape = (self.seq_length, self.image_size, self.image_size, 3)
        with open(list_path, 'r') as f:
            self.file_list = f.readlines()
        self.mean = None
        self.std = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item_list = self.file_list[idx].split(',')
        begin = int(item_list[1])
        end = begin + self.seq_length * self.step

        data_slice = np.ndarray(shape=self.input_shape, dtype=np.uint8)
        i = 0
        for j in range(begin, end, self.step):
            # e.g., images/S11_Walking.60457274_001621.jpg
            base_str = '0' * (6 - len(str(j))) + str(j) + '.jpg'
            file_name = os.path.join(self.data_root, item_list[0] + base_str)
            image = cv2.imread(file_name)
            if image.shape[0] != self.image_size:
                image = cv2.resize(image, (self.image_size, self.image_size))
            data_slice[i, :] = image
            i += 1

        # transform
        data_slice = torch.from_numpy(data_slice.transpose((0, 3, 1, 2)) / 255).float()
        data = data_slice[:self.pre_seq_length, ...]
        labels = data_slice[self.aft_seq_length:, ...]

        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=4, aft_seq_length=4, in_shape=[4, 3, 256, 256],
              distributed=False, use_prefetcher=False):

    data_root = os.path.join(data_root, 'human')
    image_size = in_shape[-1] if in_shape is not None else 256
    train_set = HumanDataset(data_root, os.path.join(data_root, 'train.txt'), image_size,
                             pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length, step=5)
    test_set = HumanDataset(data_root, os.path.join(data_root, 'test.txt'), image_size,
                            pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length, step=5)
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_test, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4,
                  use_prefetcher=True, distributed=False)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
