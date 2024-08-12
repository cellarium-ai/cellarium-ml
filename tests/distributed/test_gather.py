# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cellarium.ml.utilities.distributed import GatherLayer


def init_process(rank, size, return_dict, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, return_dict)


def run(rank: int, world_size: int, return_dict: dict) -> None:
    # Multi GPU (world_size = 2)
    # w_0       w_1
    #  |  \   /  |
    #  |    x    |
    #  |  /   \  |
    # loss_0   loss_1

    # loss_rank = coeff[rank, 0] * w[0] + coeff[rank, 1] * w[1]
    coeff = torch.tensor([[1, 2], [3, 4]])
    w_rank = torch.ones(1, requires_grad=True)  # w_rank
    w = GatherLayer.apply(w_rank)  # (w_0, w_1)
    loss = coeff[rank, 0] * w[0] + coeff[rank, 1] * w[1]
    loss.backward()
    return_dict[rank] = w.grad


def test_gather_layer():
    world_size = 2
    processes = []
    manager = mp.Manager()
    return_dict = manager.dict()
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, return_dict, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Single GPU
    # loss = coeff[0, 0] * w[0] + coeff[0, 1] * w[1] + coeff[1, 0] * w[0] + coeff[1, 1] * w[1]
    #        ---------------- rank 0 ---------------   ---------------- rank 1 ---------------
    # dloss/dw[0] = coeff[0, 0] + coeff[1, 0]
    # dloss/dw[1] = coeff[0, 1] + coeff[1, 1]
    coeff = torch.tensor([[1, 2], [3, 4]])
    w = torch.tensor([1.0, 1.0], requires_grad=True)
    loss = (coeff[0] * w).sum() + (coeff[1] * w).sum()
    loss.backward()

    for rank, w_grad in return_dict.items():
        assert w.grad is not None
        assert w_grad == w.grad[rank] == coeff[:, rank].sum()
