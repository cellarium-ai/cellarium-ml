import pytest
import torch

from scvid.data import DADCSampler, DistributedDADCSampler


@pytest.mark.parametrize("limits", [(4, 10), (2, 5, 10)])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_dadc_sampler(limits, shuffle, num_workers, batch_size):
    n_obs = limits[-1]
    sampler = DADCSampler(limits=limits, shuffle=shuffle)
    train_loader = torch.utils.data.DataLoader(
        range(n_obs),
        sampler=sampler,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    actual_idx = list(int(i) for batch in train_loader for i in batch)
    expected_idx = list(range(n_obs))

    # assert entire dataset is sampled
    assert len(expected_idx) == len(actual_idx)
    assert set(expected_idx) == set(actual_idx)

    # assert indicies are within limit bounds
    #  for lower, upper in zip((0,) + limits, limits):
    #      for idx in actual_idx[lower:upper]:
    #          assert idx >= lower and idx < upper
