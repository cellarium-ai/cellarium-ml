# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.utils._pytree import PyTree, tree_any, tree_iter, tree_map


class PyTreeDataset(torch.utils.data.Dataset):
    """
    A dataset that wraps a PyTree of tensors and ndarrays.

    Example::

        import torch
        from cellarium.ml.data import PyTreeDataset
        from cellarium.ml.utilities.data import collate_fn

        data = {
            "gene_token_nc_dict": {
                "gene_id": torch.randint(0, 10, (10, 3)),
                "gene_value": torch.randint(0, 10, (10, 3)),
            },
            "gene_token_mask_nc": torch.randint(0, 10, (10, 3)),
            "metadata_token_nc_dict": {
                "cell_type": torch.randint(0, 10, (10, 3)),
            },
            "metadata_token_mask_nc_dict": {
                "cell_type": torch.randint(0, 10, (10, 3)),
            },
            "prompt_mask_nc": torch.randint(0, 10, (10, 3)),
        }
        dataset = PyTreeDataset(data)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        for batch in dataloader:
            ...

    Args:
        pytree: A PyTree of tensors and ndarrays.
    """

    def __init__(self, pytree: PyTree) -> None:
        self._length: int = next(tree_iter(pytree)).shape[0]  # type: ignore[call-overload]
        if tree_any(lambda x: x.shape[0] != self._length, pytree):
            raise ValueError("All tensors must have the same batch dimension")
        self.pytree = pytree

    def __getitem__(self, index: int) -> PyTree:
        return tree_map(lambda data: data[index], self.pytree)

    def __getitems__(self, indices: list[int]) -> list[PyTree]:
        return [tree_map(lambda data: data[indices], self.pytree)]

    def __len__(self) -> int:
        return self._length
