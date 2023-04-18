import numpy as np
import pickle
from typing import Optional

import torch
import torch.distributed as dist

from .main_utils import get_dist_info
from .progressbar import ProgressBar


def gather_tensors(input_array):
    """Gather tensor from all GPUs."""
    world_size = dist.get_world_size()
    # gather shapes first
    myshape = input_array.shape
    mycount = input_array.size
    shape_tensor = torch.Tensor(np.array(myshape)).cuda()
    all_shape = [
        torch.Tensor(np.array(myshape)).cuda() for i in range(world_size)
    ]
    dist.all_gather(all_shape, shape_tensor)
    # compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)
    # padding tensors and gather them
    output_tensors = [
        torch.Tensor(max_count).cuda() for i in range(world_size)
    ]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = input_array.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).cuda()
    dist.all_gather(output_tensors, input_tensor)
    # unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [
        x[:all_count[i]].reshape(all_shape[i])
        for i, x in enumerate(padded_output)
    ]
    return output


def gather_tensors_batch(input_array, part_size=100, ret_rank=-1):
    """batch-wise gathering to avoid CUDA out of memory."""
    rank = dist.get_rank()
    all_features = []
    part_num = input_array.shape[0] // part_size + 1 if input_array.shape[
        0] % part_size != 0 else input_array.shape[0] // part_size
    for i in range(part_num):
        part_feat = input_array[i *
                                part_size:min((i + 1) *
                                              part_size, input_array.shape[0]),
                                ...]
        assert part_feat.shape[
            0] > 0, f'rank: {rank}, length of part features should > 0'
        gather_part_feat = gather_tensors(part_feat)
        all_features.append(gather_part_feat)
    if ret_rank == -1:
        all_features = [
            np.concatenate([all_features[i][j] for i in range(part_num)],
                           axis=0) for j in range(len(all_features[0]))
        ]
        return all_features
    else:
        if rank == ret_rank:
            all_features = [
                np.concatenate([all_features[i][j] for i in range(part_num)],
                               axis=0) for j in range(len(all_features[0]))
            ]
            return all_features
        else:
            return None


def nondist_forward_collect(func, data_loader, length, to_numpy=False):
    """Forward and collect network outputs.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a list of CPU tensors.
        length (int): Expected length of output arrays.
        to_numpy (bool): Whether to conver tensors to the numpy array.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    """
    results = []
    prog_bar = ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(*data)  # list{tensor, ...}
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        if to_numpy:
            results_all[k] = np.concatenate(
                [batch[k].cpu().numpy() for batch in results], axis=0)
        else:
            results_all[k] = torch.cat(
                [batch[k] for batch in results], dim=0)
        assert results_all[k].shape[0] == length
    return results_all


def dist_forward_collect(func, data_loader, rank, length, ret_rank=-1, to_numpy=False):
    """Forward and collect network outputs in a distributed manner.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a list of CPU tensors.
        rank (int): This process id.
        length (int): Expected length of output arrays.
        ret_rank (int): The process that returns.
            Other processes will return None.
        to_numpy (bool): Whether to conver tensors to the numpy array.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    """
    assert to_numpy == True
    results = []
    if rank == 0:
        prog_bar = ProgressBar(len(data_loader))
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(*data)  # list{tensor, ...}
        results.append(result)

        if rank == 0:
            prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_cat = np.concatenate([batch[k].cpu().numpy() for batch in results],
                                     axis=0)
        if ret_rank == -1:
            results_gathered = gather_tensors_batch(results_cat, part_size=20)
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
        else:
            results_gathered = gather_tensors_batch(
                results_cat, part_size=20, ret_rank=ret_rank)
            if rank == ret_rank:
                results_strip = np.concatenate(
                    results_gathered, axis=0)[:length]
            else:
                results_strip = None
        results_all[k] = results_strip
    return results_all


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None
