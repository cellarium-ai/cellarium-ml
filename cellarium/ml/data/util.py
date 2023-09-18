# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info as _get_worker_info


def get_rank_and_num_replicas() -> tuple[int, int]:
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
        raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas-1}]")
    return rank, num_replicas


def get_worker_info() -> tuple[int, int]:
    """
    This helper function returns ``worker_id`` and ``num_workers``. If it is running
    in the main process then it returns ``worker_id=0`` and ``num_workers=1``.
    """
    worker_info = _get_worker_info()
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
    return worker_id, num_workers


def collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray | torch.Tensor]:
    """
    Collate function for the ``DataLoader``. This function assumes that the batch is a list of
    dictionaries, where each dictionary has the same keys. The values of each key are converted
    to a :class:`torch.Tensor` and concatenated along the first dimension. If the key is ``obs_names``,
    the values are concatenated along the first dimension without converting to a :class:`torch.Tensor`.
    """
    keys = batch[0].keys()
    collated_batch = {}
    if len(batch) > 1:
        assert all(keys == data.keys() for data in batch[1:]), "All dictionaries in the batch must have the same keys."
    for key in keys:
        if key == "obs_names":
            collated_batch[key] = np.concatenate([data[key] for data in batch], axis=0)
        elif key == "var_names":
            # Check that all var_names are the same
            if len(batch) > 1:
                assert all(
                    np.array_equal(batch[0][key], data[key]) for data in batch[1:]
                ), "All dictionaries in the batch must have the same var_names."
            # If so, just take the first one
            collated_batch[key] = batch[0][key]
        else:
            collated_batch[key] = torch.cat([torch.from_numpy(data[key]) for data in batch], dim=0)
    return collated_batch
