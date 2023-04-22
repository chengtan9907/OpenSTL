import argparse
import cv2
import os
import errno
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from shutil import copyfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='data/human',
                        help='path to the original images')
    parser.add_argument('--src_name', type=str, default=None,
                        help='the source folder name')
    parser.add_argument('--dst_name', type=str, default=None,
                        help='the destination folder name')
    parser.add_argument('--shape', type=int,
                        nargs='+', default=[256, 256],
                        help='target image size of resizing')
    args = parser.parse_args()
    return args


def resize_image(inputFileName, output_size=[256, 256], input_str='images', output_str='images_resized'):
    try:
        out_path = inputFileName.replace(input_str, output_str)

        if not os.path.exists(os.path.dirname(out_path)):
            try:
                os.makedirs(os.path.dirname(out_path))
            except OSError as exc:  # Guard against race condition
                print("OSError ",inputFileName)
                if exc.errno != errno.EEXIST:
                    raise

        assert out_path != inputFileName
        im = cv2.imread(inputFileName)
        shape = im.shape
        max_dim_in = max(shape[0], shape[1])
        max_dim_ou = max(output_size[0], output_size[1])
        if max_dim_in < max_dim_ou:
            copyfile(inputFileName, out_path)
        else:
            im_resize = cv2.resize(im, (output_size[0], output_size[1]))
            cv2.imwrite(out_path, im_resize)
    except:
        print("general failure ",inputFileName)


def main():
    args = parse_args()
    src_name = "images" if args.src_name is None else args.src_name
    dst_name = "images_resized" if args.dst_name is None else args.dst_name
    base_dir = os.path.join(args.path, src_name)

    print("scanning files...")
    files = [y for x in os.walk(base_dir) for y in glob(os.path.join(x[0], '*.*'))]
    print("total files {}, start resizing...".format(len(files)))

    pool = ThreadPool(8)
    resize_image_fun = partial(resize_image, input_str=src_name, output_str=dst_name)
    pool.map(resize_image_fun, files)


if __name__ == '__main__':
    main()
