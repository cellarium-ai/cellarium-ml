from scvid.data import DistributedAnnDataCollectionSampler


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
