import argparse
import cv2
import random
import numpy as np

from openstl.datasets.dataloader_moving_mnist import MovingMNIST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', type=str, default='fmnist',
                        help='name of MNIST variants to generate')
    args = parser.parse_args()
    return args


def generate_mmnist_testset(root, data_name='mnist', num_objects=[2], save_path=None, length=10000):
    dataset = MovingMNIST(root=root, data_name=data_name, is_train=True, num_objects=num_objects)
    test_list = []

    for i in range(length):
        num_digits = random.choice(dataset.num_objects)
        images = dataset.generate_moving_mnist(num_digits, dataset.background)
        if not dataset.background:
            test_list.append(images.reshape(20, 1, 64, 64))
        else:
            test_list.append(images.reshape(20, 1, 64, 64, 3))
    test_data = np.concatenate(test_list, axis=1)
    np.save(save_path, test_data)

    return test_data


def main():
    # fix randomness
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # generate the test set of Moving MMNIST variants
    args = parse_args()
    root = 'data/'
    num_objects = [2]
    file_map = {
        'mnist': 'data/moving_mnist/mnist_test_seq.npy',
        'fmnist': 'data/moving_fmnist/fmnist_test_seq.npy',
        'mnist_cifar': 'data/moving_mnist/mnist_cifar_test_seq.npy',
    }

    data_name = args.data_name  # 'fmnist'
    save_path = file_map[data_name]
    length = 10000

    test_data = generate_mmnist_testset(root, data_name, num_objects, save_path, length)
    print('generate test set:', test_data.shape)
    cv2.imwrite(f'tools/prepare_data/{data_name}_sample.jpg', test_data[0, 5, ...])


if __name__ == '__main__':
    main()
