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


def padding_transpose_conv2d_layers(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
):
    """
    Determining correct padding of transpose 2D convolutional layers. Note the input and output shape of layers must
    be known.

    :param input_shape: input shape of Conv2D layer
    :param output_shape: output shape of Conv2D layer
    :param kernel_size: kernel size of Conv2D layer
    :param stride: stride of Conv2D layer
    """
    padding = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        pad = (
            (input_shape[idx] - 1) * stride[idx] - output_shape[idx] + kernel_size[idx]
        ) / 2
        padding[idx] = int(pad)

    return tuple(padding)


def min_max_normalization(x: np.array, eps: float = 1e-10) -> np.array:
    """
    Min-max normalize a numpy array, i.e. values are in the range [0, 1].

    :param x: numpy array of arbitrary shape
    :param eps: Float to avoid division by zeros
    :return: min-max normalized input array
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)


def z_score_normalization(x: np.array) -> np.array:
    """
    Normalize data by z-score
    .. math::
        \frax{x - \mu}{\sigma}

    :param x: numpy array of arbitrary shape
    :return: normalized input array using z-score
    """
    return (x - np.mean(x)) / np.std(x)
