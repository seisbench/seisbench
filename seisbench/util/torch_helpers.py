import numpy as np
import torch


def worker_seeding(wid):
    """
    When using numpy random inside multiple workers in the data loader, they all produce the same random numbers,
    as the seed is shared. As a solution, the worker init function can be overwritten. This solution uses the torch
    initial_seed, which is set separately for each worker.
    This should be taken into account, as SeisBench uses numpy random for augmentation.

    To set the seed in each worker, use `worker_init_fn=worker_seeding` when creating the pytorch DataLoader.

    Code from https://github.com/pytorch/pytorch/issues/5059

    :param wid: Worker id
    :type wid: int
    """
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xFFFF_FFFF])


def output_shape_conv2d_layers(
    input_shape: tuple[int, int],
    padding: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
):
    """
    Determining correct output shape for Conv2D layers in PyTorch.

    :param input_shape: input shape of Conv2D layer
    :param padding: padding of Conv2D layer
    :param kernel_size: kernel size of Conv2D layer
    :param stride: stride of Conv2D layer
    """
    output_shape = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        out = (input_shape[idx] + 2 * padding[idx] - kernel_size[idx]) / stride[idx] + 1
        output_shape[idx] = int(out)

    return tuple(output_shape)
