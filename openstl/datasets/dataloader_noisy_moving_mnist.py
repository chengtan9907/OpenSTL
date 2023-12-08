import numpy as np
import os
import random

from openstl.datasets.utils import create_loader
from openstl.datasets.dataloader_moving_mnist import MovingMNIST


class NoisyMovingMNIST(MovingMNIST):
    """Noisy Moving MNIST Dataset`_

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
    def __init__(self, root, is_train=True, data_name='mnist', n_frames_input=10, n_frames_output=10, image_size=64, num_objects=..., sigma=0.2, mis_prob=0.2, noise_type='perceptual', transform=None, use_augment=False):
        # only support mnist
        data_name = 'mnist'
        self.sigma = sigma
        self.mis_prob = mis_prob
        self.noise_type = noise_type
        MovingMNIST.__init__(self, root, is_train, data_name, n_frames_input, n_frames_output, image_size, num_objects, transform, use_augment)

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi

        if self.noise_type == 'dynamic':
            v_ys = (np.random.normal(0, self.sigma, self.n_frames_input) + np.sin(theta)).tolist() + [np.sin(theta)] * self.n_frames_output
            v_xs = (np.random.normal(0, self.sigma, self.n_frames_input) + np.cos(theta)).tolist() + [np.cos(theta)] * self.n_frames_output
        else:
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
        data = np.zeros((self.n_frames_total, self.image_size_,
                        self.image_size_), dtype=np.float32)

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

        if self.noise_type == 'perceptual':
            for i in range(self.n_frames_input):
                pos_x = np.random.randint(0, self.image_size_ - 24)
                pos_y = np.random.randint(0, self.image_size_ - 24)
                data[i, pos_x:pos_x+24, pos_y:pos_y+24] = 0
        elif self.noise_type == 'missing':
            mis_idx = np.random.choice([0, 1], size=self.n_frames_input, p=[1-self.mis_prob, self.mis_prob])
            mis_idx = np.where(mis_idx == 1)[0]
            data[:self.n_frames_input][mis_idx] = 0

        data = data[..., np.newaxis]
        return data


def load_data(batch_size, val_batch_size, data_root, num_workers=4, data_name='mnist',
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 64, 64],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False, noise_type='perceptual'):

    image_size = in_shape[-1] if in_shape is not None else 64
    train_set = NoisyMovingMNIST(root=data_root, is_train=True, data_name=data_name,
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length, num_objects=[2],
                            image_size=image_size, use_augment=use_augment, noise_type=noise_type)
    test_set = NoisyMovingMNIST(root=data_root, is_train=False, data_name=data_name,
                           n_frames_input=pre_seq_length,
                           n_frames_output=aft_seq_length, num_objects=[2],
                           image_size=image_size, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=False, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=False, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=False, drop_last=drop_last,
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
