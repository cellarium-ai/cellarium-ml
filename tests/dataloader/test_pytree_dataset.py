# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch.utils._pytree import tree_iter

from cellarium.ml.data import PyTreeDataset
from cellarium.ml.utilities.data import collate_fn


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_pytree_dataset(batch_size):
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
        sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    list_batch = list(dataloader)
    full_batch = collate_fn(list_batch)
    for x1, x2 in zip(tree_iter(data), tree_iter(full_batch)):
        assert torch.equal(x1, x2)
