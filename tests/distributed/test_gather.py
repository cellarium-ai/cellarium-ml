import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cellarium.ml.distributed import GatherLayer


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

    # loss_0 = w_0 + w_1
    # loss_1 = w_0 + w_1

    # dloss/dw_0 = dloss_0/dw_0 + dloss_1/dw_0 = 2
    # dloss/dw_1 = dloss_0/dw_1 + dloss_1/dw_1 = 2
    w = torch.ones(1, requires_grad=True)  # w_rank
    gathered_w = GatherLayer.apply(w)  # (w_0, w_1)
    loss = gathered_w[0] + gathered_w[1]  # w_0 + w_1
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
    # loss = w + w
    # dloss/dw = 2
    w = torch.ones(1, requires_grad=True)
    gathered_w = (w, w)
    loss = gathered_w[0] + gathered_w[1]  # w + w
    loss.backward()

    for w_grad in return_dict.values():
        assert w_grad == w.grad
