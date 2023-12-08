import cv2
import gzip
import numpy as np
import os
import random

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


def load_cifar(root, data_name='mnist_cifar'):
    # Load CIFAR-10 dataset as the background.
    data = None
    if 'cifar' in data_name:
        path = os.path.join(root, 'cifar10')
        cifar_train = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
        cifar_test = torchvision.datasets.CIFAR10(root=path, train=False, download=True)
        data = np.concatenate([cifar_train.data, cifar_test.data],
                              axis=0).reshape(-1, 32, 32, 3)
    return data


def load_mnist(root, data_name='mnist'):
    # Load MNIST dataset for generating training data.
    file_map = {
        'mnist': 'moving_mnist/train-images-idx3-ubyte.gz',
        'fmnist': 'moving_fmnist/train-images-idx3-ubyte.gz',
        'mnist_cifar': 'moving_mnist/train-images-idx3-ubyte.gz',
    }
    path = os.path.join(root, file_map[data_name])
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root, data_name='mnist'):
    # Load the fixed dataset
    file_map = {
        'mnist': 'moving_mnist/mnist_test_seq.npy',
        'fmnist': 'moving_fmnist/fmnist_test_seq.npy',
        'mnist_cifar': 'moving_mnist/mnist_cifar_test_seq.npy',
    }
    path = os.path.join(root, file_map[data_name])
    dataset = np.load(path)
    if 'cifar' not in data_name:
        dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(Dataset):
    """Moving MNIST Dataset <http://arxiv.org/abs/1502.04681>`_

    Args:
        data_root (str): Path to the dataset.
        is_train (bool): Whether to use the train or test set.
        data_name (str): Name of the MNIST modality.
        n_frames_input, n_frames_output (int): The number of input and prediction
            video frames.
        image_size (int): Input resolution of the data.
        num_objects (list): The number of moving objects in videos.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, root, is_train=True, data_name='mnist',
                 n_frames_input=10, n_frames_output=10, image_size=64,
                 num_objects=[2], transform=None, use_augment=False):
        super(MovingMNIST, self).__init__()

        self.dataset = None
        self.is_train = is_train
        self.data_name = data_name
        if self.is_train:
            self.mnist = load_mnist(root, data_name)
            self.cifar = load_cifar(root, data_name)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root, data_name)
                self.cifar = load_cifar(root, data_name)
            else:
                self.dataset = load_fixed_set(root, data_name)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        self.use_augment = use_augment
        self.background = 'cifar' in data_name
        # For generating data
        self.image_size_ = image_size
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi

        v_ys = [np.sin(theta)] * seq_length
        v_xs = [np.cos(theta)] * seq_length

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        bounce_x = 1
        bounce_y = 1
        for i, v_x, v_y in zip(range(seq_length), v_xs, v_ys):
            # Take a step along velocity.
            y += bounce_y * v_y * self.step_length_
            x += bounce_x * v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                # v_x = -v_x
                bounce_x = -bounce_x
            if x >= 1.0:
                x = 1.0
                # v_x = -v_x
                bounce_x = -bounce_x
            if y <= 0:
                y = 0
                # v_y = -v_y
                bounce_y = -bounce_y
            if y >= 1.0:
                y = 1.0
                # v_y = -v_y
                bounce_y = -bounce_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2, background=False):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        if not background:  # `black`
            data = np.zeros((self.n_frames_total, self.image_size_,
                            self.image_size_), dtype=np.float32)
        else:  # cifar-10 as the background
            ind = random.randint(0, self.cifar.shape[0] - 1)
            back = cv2.resize(self.cifar[ind], (self.image_size_, self.image_size_), interpolation=cv2.INTER_CUBIC)
            data = np.repeat(back[np.newaxis, ...], self.n_frames_total, axis=0).astype(np.uint8)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind].copy()
            if background:  # binary {0, 255}
                digit_image[digit_image > 1] = 255
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                if not background:
                    data[i, top:bottom, left:right] = np.maximum(
                        data[i, top:bottom, left:right], digit_image)
                else:
                    data[i, top:bottom, left:right, ...] = np.maximum(
                        data[i, top:bottom, left:right, ...], np.repeat(digit_image[..., np.newaxis], 3, axis=2))

        if not background:
            data = data[..., np.newaxis]
        return data

    def _augment_seq(self, imgs, crop_scale=0.94):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [10, 1, 64, 64]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(-2, 1):
            imgs = torch.flip(imgs, dims=(2,3))  # rotation 180
        elif random.randint(-2, 1):
            imgs = torch.flip(imgs, dims=(2, ))  # vertical flip
        elif random.randint(-2, 1):
            imgs = torch.flip(imgs, dims=(3, ))  # horizontal flip
        return imgs

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits, self.background)
        else:
            images = self.dataset[:, idx, ...]

        if not self.background:
            r, w = 1, self.image_size_
            images = images.reshape((length, w, r, w, r)).transpose(
                0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        else:
            images = images.transpose(0, 3, 1, 2)

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()

        if self.use_augment:
            imgs = self._augment_seq(torch.cat([input, output], dim=0), crop_scale=0.94)
            input = imgs[:self.n_frames_input, ...]
            output = imgs[self.n_frames_input:self.n_frames_input+self.n_frames_output, ...]

        return input, output

    def __len__(self):
        return self.length


def load_data(batch_size, val_batch_size, data_root, num_workers=4, data_name='mnist',
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 64, 64],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    image_size = in_shape[-1] if in_shape is not None else 64
    train_set = MovingMNIST(root=data_root, is_train=True, data_name=data_name,
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length, num_objects=[2],
                            image_size=image_size, use_augment=use_augment)
    test_set = MovingMNIST(root=data_root, is_train=False, data_name=data_name,
                           n_frames_input=pre_seq_length,
                           n_frames_output=aft_seq_length, num_objects=[2],
                           image_size=image_size, use_augment=False)

    dataloader_train = create_loader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True, is_training=True,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, persistent_workers=True,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
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
                  data_name='mnist',
                  pre_seq_length=10, aft_seq_length=10,
                  distributed=True, use_prefetcher=False)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
