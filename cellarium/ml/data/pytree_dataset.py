# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.utils._pytree import PyTree, tree_any, tree_iter, tree_map


class PyTreeDataset(torch.utils.data.Dataset):
    """
    A dataset that wraps a PyTree of tensors or ndarrays.

    Args:
        pytree: A PyTree of tensors or ndarrays.
    """

    def __init__(self, pytree: PyTree) -> None:
        self._length = next(tree_iter(pytree)).shape[0]  # type: ignore[call-overload]
        if tree_any(lambda x: x.shape[0] != self._length, pytree):
            raise ValueError("All tensors must have the same batch dimension")
        self.pytree = pytree

    def __getitem__(self, index: int) -> PyTree:
        return tree_map(lambda x: x[index], self.pytree)

    def __getitems__(self, indices: list[int]) -> list[PyTree]:
        return [tree_map(lambda x: x[indices], self.pytree)]

    def __len__(self) -> int:
        return self._length
