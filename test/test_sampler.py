import os

import numpy as np
import pytest
import torch
from anndata import AnnData

from scvid.data import (
    DistributedAnnDataCollection,
    DistributedAnnDataCollectionDataset,
    DistributedAnnDataCollectionSampler,
    DistributedAnnDataCollectionSingleConsumerSampler,
    collate_fn,
)


@pytest.mark.parametrize("limits", [(4, 10), (2, 5, 10)])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_dadc_sampler_indices(limits, shuffle, num_workers, batch_size):
    n_obs = limits[-1]
    sampler = DistributedAnnDataCollectionSingleConsumerSampler(
        limits=limits, shuffle=shuffle
    )
    data_loader = torch.utils.data.DataLoader(
        range(n_obs),
        sampler=sampler,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    actual_idx = list(int(i) for batch in data_loader for i in batch)
    expected_idx = list(range(n_obs))

    # assert entire dataset is sampled
    assert len(expected_idx) == len(actual_idx)
    assert set(expected_idx) == set(actual_idx)


@pytest.fixture
def dadc(tmp_path):
    n_cell, g_gene = (10, 5)
    limits = [2, 5, 10]

    rng = np.random.default_rng(1465)
    X = rng.integers(10, size=(n_cell, g_gene))
    adata = AnnData(X)
    for i, limit in enumerate(zip([0] + limits, limits)):
        sliced_adata = adata[slice(*limit)]
        sliced_adata.write(os.path.join(tmp_path, f"adata.00{i}.h5ad"))

    # distributed anndata
    filenames = str(os.path.join(tmp_path, "adata.{000..002}.h5ad"))
    dadc = DistributedAnnDataCollection(
        filenames,
        limits,
        max_cache_size=1,
        cache_size_strictly_enforced=True,
    )
    # clear cache
    dadc.cache.clear()
    return dadc


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_dadc_sampler_misses(dadc, shuffle, num_workers, batch_size):
    dataset = DistributedAnnDataCollectionDataset(dadc)
    sampler = DistributedAnnDataCollectionSingleConsumerSampler(
        limits=dadc.limits, shuffle=shuffle
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    # iterate through dataloader
    list(data_loader)

    # assert one epoch
    assert sampler.epoch == 1

    if num_workers > 0:
        # workers create their own instance of the dataset
        assert dadc.cache.miss_count == 0
    else:
        # each anndata was loaded only once
        assert dadc.cache.miss_count == len(dadc.limits)


def test_shards_basic():
    num_shards = 5
    shard_side = 100
    num_replicas = 2
    seed = 0

    s0 = DistributedAnnDataCollectionSampler(
        num_shards=num_shards,
        shard_size=shard_side,
        seed=seed,
        num_replicas=num_replicas,
        rank=0,
    )

    s1 = DistributedAnnDataCollectionSampler(
        num_shards=num_shards,
        shard_size=shard_side,
        seed=seed,
        num_replicas=num_replicas,
        rank=1,
    )

    # each process should get the same number of shards
    # so even though 5 shards were specified, only 4 should be used
    assert len(s0.process_shard_indexes) == 2
    assert len(s1.process_shard_indexes) == 2

    # and the shards shoud be disjoint
    assert len(set(s0.process_shard_indexes + s1.process_shard_indexes)) == 4


def test_shards_iter():
    shard_size = 100
    s0 = DistributedAnnDataCollectionSampler(
        num_shards=4, shard_size=shard_size, seed=0, num_replicas=2, rank=0
    )

    # calculate the offsets we expect
    expected = []
    for offset in [shard_size * i for i in s0.process_shard_indexes]:
        for x in range(0, shard_size):
            expected.append(offset + x)

    actual = list(s0)

    expected_length = len(s0.process_shard_indexes) * shard_size
    assert len(expected) == expected_length
    assert len(actual) == expected_length

    assert len(set(expected + actual)) == expected_length
