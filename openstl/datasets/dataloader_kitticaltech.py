import os
import os.path as osp
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

from .utils import create_loader

try:
    import hickle as hkl
except ImportError:
    hkl = None


# cite the `process_im` code from PredNet, Thanks!
# https://github.com/coxlab/prednet/blob/master/process_kitti.py
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = resize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))), preserve_range=True)
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


class KittiCaltechDataset(Dataset):
    """KittiCaltech <https://dl.acm.org/doi/10.1177/0278364913491297>`_ Dataset"""

    def __init__(self, datas, indices, pre_seq_length, aft_seq_length, require_back=False):
        super(KittiCaltechDataset, self).__init__()
        self.datas = datas.swapaxes(2, 3).swapaxes(1, 2)
        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.require_back = require_back
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = end1 + self.aft_seq_length
        data = torch.tensor(self.datas[begin:end1, ::]).float()
        labels = torch.tensor(self.datas[end1:end2, ::]).float()
        return data, labels


class DataProcess(object):
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.seq_len = input_param['seq_length']

    def load_data(self, mode='train'):
        """Loads the dataset.
        Args:
          paths: paths of train/test dataset.
          mode: Training or testing.
        Returns:
          A dataset and indices of the sequence.
        """
        if mode == 'train' or mode == 'val':
            kitti_root = self.paths['kitti']
            data = hkl.load(osp.join(kitti_root, 'X_' + mode + '.hkl'))
            data = data.astype('float') / 255.0
            fileidx = hkl.load(
                osp.join(kitti_root, 'sources_' + mode + '.hkl'))

            indices = []
            index = len(fileidx) - 1
            while index >= self.seq_len - 1:
                if fileidx[index] == fileidx[index - self.seq_len + 1]:
                    indices.append(index - self.seq_len + 1)
                    index -= self.seq_len - 1
                index -= 1

        elif mode == 'test':
            caltech_root = self.paths['caltech']
            data = []
            fileidx = []
            for seq_id in os.listdir(caltech_root):
                if osp.isdir(osp.join(caltech_root, seq_id)) is False:
                    continue
                for item in os.listdir(osp.join(caltech_root, seq_id)):
                    cap = cv2.VideoCapture(
                        osp.join(caltech_root, seq_id, item))
                    cnt_frames = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        cnt_frames += 1
                        if cnt_frames % 3 == 0:
                            frame = process_im(frame, (128, 160)) / 255.0
                            data.append(frame)
                            fileidx.append(seq_id + item)
            data = np.asarray(data)

            indices = []
            index = len(fileidx) - 1
            while index >= self.seq_len - 1:
                if fileidx[index] == fileidx[index - self.seq_len + 1]:
                    indices.append(index - self.seq_len + 1)
                    index -= self.seq_len - 1
                index -= 1

        return data, indices


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=1, distributed=False):

    if os.path.exists(osp.join(data_root, 'kitti_hkl')):
        input_param = {
            'paths': {'kitti': osp.join(data_root, 'kitti_hkl'),
                    'caltech': osp.join(data_root, 'caltech')},
            'seq_length': (pre_seq_length + aft_seq_length),
            'input_data_type': 'float32',
        }
        input_handle = DataProcess(input_param)
        train_data, train_idx = input_handle.load_data('train')
        test_data, test_idx = input_handle.load_data('test')
    elif os.path.exists(osp.join(data_root, 'kitticaltech_npy')):
        train_data = np.load(osp.join(data_root, 'kitticaltech_npy', 'train_data.npy'))
        train_idx = np.load(osp.join(data_root, 'kitticaltech_npy', 'train_idx.npy'))
        test_data = np.load(osp.join(data_root, 'kitticaltech_npy', 'test_data.npy'))
        test_idx = np.load(osp.join(data_root, 'kitticaltech_npy', 'test_idx.npy'))
    else:
        assert False and "Invalid data_root for kitticaltech dataset"

    train_set = KittiCaltechDataset(
        train_data, train_idx, pre_seq_length, aft_seq_length)
    test_set = KittiCaltechDataset(
        test_data, test_idx, pre_seq_length, aft_seq_length)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers, distributed=distributed)
    dataloader_vali = None
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed)

    return dataloader_train, dataloader_vali, dataloader_test
