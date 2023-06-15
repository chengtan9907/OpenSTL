import argparse
import os
import numpy as np

from openstl.datasets import dataset_parameters
from openstl.utils import (show_video_gif_multiple, show_video_gif_single, show_video_line,
                           show_taxibj, show_weather_bench)


def min_max_norm(data):
    _min, _max = np.min(data), np.max(data)
    data = (data - _min) / (_max - _min)
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualization of a STL model')

    parser.add_argument('--dataname', '-d', default=None, type=str,
                        help='The name of dataset (default: "mmnist")')
    parser.add_argument('--index', '-i', default=0, type=int, help='The index of a video sequence to show')
    parser.add_argument('--work_dirs', '-w', default=None, type=str,
                        help='Path to the work_dir or the path to a set of work_dirs')
    parser.add_argument('--vis_dirs', '-v', action='store_true', default=False,
                        help='Whether to visualize a set of work_dirs')
    parser.add_argument('--reload_input', action='store_true', default=False,
                        help='Whether to reload the input and true for each method')
    parser.add_argument('--save_dirs', '-s', default='vis_figures', type=str,
                        help='The path to save visualization results')
    parser.add_argument('--vis_channel', '-vc', default=-1, type=int,
                        help='Select a channel to visualize as the heatmap')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.dataname is not None and args.work_dirs is not None, \
        'The name of dataset and the path to work_dirs are required'

    # setup results of the STL methods
    base_dir = args.work_dirs
    assert os.path.isdir(args.work_dirs)
    if args.vis_dirs:
        method_list = os.listdir(args.work_dirs)
    else:
        method_list = [args.work_dirs.split('/')[-1]]
        base_dir = base_dir.split(method_list[0])[0]

    use_rgb = False if args.dataname in ['mfmnist', 'mmnist', 'kth20', 'kth', 'kth40'] else True
    config = args.__dict__
    config.update(dataset_parameters[args.dataname])
    idx, ncols = args.index, config['aft_seq_length']
    if not os.path.isdir(args.save_dirs):
        os.mkdir(args.save_dirs)
    if args.vis_channel != -1:  # choose a channel
        c_surfix = f"_C{args.vis_channel}"
        assert 0 <= args.vis_channel <= config['in_shape'][1], 'Channel index out of range'
    else:
        c_surfix = ""
        assert args.dataname not in ['taxibj', 'weather_uv10_5_625'], 'Please select a channel'

    # loading results
    predicts_dict, inputs_dict, trues_dict = dict(), dict(), dict()
    empty_keys = list()
    for method in method_list:
        try:
            predicts_dict[method] = np.load(os.path.join(base_dir, method, 'saved/preds.npy'))
            if 'weather' in args.dataname:
                predicts_dict[method] = min_max_norm(predicts_dict[method])
        except:
            empty_keys.append(method)
            print('Failed to read the results of', method)
    assert len(predicts_dict.keys()) >= 1, 'The results should not be empty'
    for k in empty_keys:
        method_list.pop(method_list.index(k))

    for method in method_list:
        inputs = np.load(os.path.join(base_dir, method_list[0], 'saved/inputs.npy'))
        trues = np.load(os.path.join(base_dir, method_list[0], 'saved/trues.npy'))
        if 'weather' in args.dataname:
            inputs = min_max_norm(inputs)
            trues = min_max_norm(trues)
            inputs = show_weather_bench(inputs[idx, 0:ncols, ...], src_img=None, cmap='GnBu').transpose(0, 3, 1, 2)
            trues = show_weather_bench(trues[idx, 0:ncols, ...], src_img=None, cmap='GnBu').transpose(0, 3, 1, 2)
        elif 'taxibj' in args.dataname:
            inputs = show_taxibj(inputs[idx, 0:ncols, ...], cmap='viridis').transpose(0, 3, 1, 2)
            trues = show_taxibj(trues[idx, 0:ncols, ...], cmap='viridis').transpose(0, 3, 1, 2)
        else:
            inputs, trues = inputs[idx], trues[idx]
        if not args.reload_input:  # load the input and true for each method
            break
        else:
            inputs_dict[method], trues_dict[method] = inputs, trues

    # plot gifs and figures of the STL methods
    for i, method in enumerate(method_list):
        print(method, predicts_dict[method][idx].shape)
        if args.reload_input:
            inputs, trues = inputs_dict[method], trues_dict[method]
        if 'weather' in args.dataname:
            preds = show_weather_bench(predicts_dict[method][idx, 0:ncols, ...],
                                       src_img=None, cmap='GnBu', vis_channel=args.vis_channel)
            preds = preds.transpose(0, 3, 1, 2)
        elif 'taxibj' in args.dataname:
            preds = show_taxibj(predicts_dict[method][idx, 0:ncols, ...],
                                cmap='viridis', vis_channel=args.vis_channel)
            preds = preds.transpose(0, 3, 1, 2)
        else:
            preds = predicts_dict[method][idx]

        if i == 0:
            show_video_line(inputs.copy(), ncols=config['pre_seq_length'], vmax=0.6, cbar=False,
                out_path='{}/{}_input{}'.format(args.save_dirs, args.dataname+c_surfix, str(idx)+'.png'),
                format='png', use_rgb=use_rgb)
            show_video_line(trues.copy(), ncols=config['aft_seq_length'], vmax=0.6, cbar=False,
                out_path='{}/{}_true{}'.format(args.save_dirs, args.dataname+c_surfix, str(idx)+'.png'),
                format='png', use_rgb=use_rgb)
            show_video_gif_single(inputs.copy(), use_rgb=use_rgb,
                out_path='{}/{}_{}_{}_input'.format(args.save_dirs, args.dataname+c_surfix, method, idx))
            show_video_gif_single(trues.copy(), use_rgb=use_rgb,
                out_path='{}/{}_{}_{}_true'.format(args.save_dirs, args.dataname+c_surfix, method, idx))

        show_video_line(preds, ncols=ncols, vmax=0.6, cbar=False,
                        out_path='{}/{}_{}_{}'.format(args.save_dirs, args.dataname+c_surfix, method, str(idx)+'.png'),
                        format='png', use_rgb=use_rgb)
        show_video_gif_multiple(inputs, trues, preds, use_rgb=use_rgb,
                                out_path='{}/{}_{}_{}'.format(args.save_dirs, args.dataname+c_surfix, method, idx))
        show_video_gif_single(preds, use_rgb=use_rgb,
                              out_path='{}/{}_{}_{}_pred'.format(args.save_dirs, args.dataname+c_surfix, method, idx))


if __name__ == '__main__':
    main()
