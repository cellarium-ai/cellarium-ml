import warnings
from typing import Tuple

import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info as _get_worker_info


def get_rank_and_num_replicas() -> Tuple[int, int]:
    """
    This helper function returns the rank of the current process and
    the number of processes in the default process group. If distributed
    package is not available or default process group has not been initialized
    then it returns ``rank=0`` and ``num_replicas=1``.
    """
    if not dist.is_available():
        num_replicas = 1
        rank = 0
    else:
        try:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        except RuntimeError:
            warnings.warn(
                "Distributed package is available but the default process group has not been initialized. "
                "Falling back to ``rank=0`` and ``num_replicas=1``.",
                UserWarning,
            )
            num_replicas = 1
            rank = 0
    if rank >= num_replicas or rank < 0:
        raise ValueError(
            f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas-1}]"
        )
    return rank, num_replicas


def get_worker_info() -> Tuple[int, int]:
    """
    This helper function always returns ``worker_id`` and ``num_workers``
    even if ``worker_info`` is ``None``.
    """
    worker_info = _get_worker_info()
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
    return worker_id, num_workers


def collate_fn(batch):
    keys = batch[0].keys()
    return {
        key: torch.cat([torch.from_numpy(data[key]) for data in batch], dim=0)
        for key in keys
    }
