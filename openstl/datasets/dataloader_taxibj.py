import torch
import numpy as np
from torch.utils.data import Dataset

from .utils import create_loader


class TaxibjDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, X, Y):
        super(TaxibjDataset, self).__init__()
        self.X = (X+1)/2
        self.Y = (Y+1)/2
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=None, aft_seq_length=None, distributed=False):

    dataset = np.load(data_root+'taxibj/dataset.npz')
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset[
        'Y_train'], dataset['X_test'], dataset['Y_test']
    train_set = TaxibjDataset(X=X_train, Y=Y_train)
    test_set = TaxibjDataset(X=X_test, Y=Y_test)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers, distributed=distributed)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed)

    return dataloader_train, dataloader_vali, dataloader_test
