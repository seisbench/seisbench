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
