# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from cellarium.ml.api import _cnmf_consensus as cc


def planted_replicates(r: int = 12, k: int = 3, g: int = 40, noise: float = 0.05, seed: int = 0):
    """The factors of ``r`` replicate runs that recover ``k`` programs, in a random order in each replicate."""
    generator = torch.Generator().manual_seed(seed)
    programs_kg = torch.rand(k, g, generator=generator) ** 3
    replicates = []
    for _ in range(r):
        factors_kg = programs_kg + noise * torch.rand(k, g, generator=generator)
        replicates.append(factors_kg[torch.randperm(k, generator=generator)])
    return torch.stack(replicates), programs_kg


def test_consensus_recovers_the_planted_programs():
    D_rkg, programs_kg = planted_replicates()
    result = cc.consensus_factors(D_rkg, density_threshold=0.5, local_neighborhood_size=0.3)

    consensus_kg = result["consensus_D_kg"]
    assert consensus_kg.shape == (3, 40)
    assert torch.allclose(consensus_kg.sum(dim=-1), torch.ones(3))  # L1-normalized
    # every planted program is the closest consensus factor to exactly one consensus factor (up to the order)
    planted_kg = torch.nn.functional.normalize(programs_kg, p=1, dim=-1)
    nearest = torch.cdist(planted_kg, consensus_kg).argmin(dim=1)
    assert sorted(nearest.tolist()) == [0, 1, 2]
    assert result["stability"] > 0.8
    assert result["n_neighbors"] == int(12 * 0.3)
    assert result["all_neighbor_distances"].shape == (12 * 3, result["n_neighbors"])
    assert result["density_threshold"] == 0.5
    assert result["local_neighborhood_size"] == 0.3


def test_consensus_does_not_hold_the_quadratic_distance_matrix():
    D_rkg, _ = planted_replicates()
    result = cc.consensus_factors(D_rkg, density_threshold=0.5, local_neighborhood_size=0.3)
    assert "filtered_euclidean_distance_matrix" not in result
    assert not any(isinstance(v, torch.Tensor) and v.shape == (36, 36) for v in result.values())


def test_filtered_distance_matrix_is_recomputed_on_demand():
    D_rkg, _ = planted_replicates()
    matrix_ff = cc.filtered_distance_matrix(D_rkg, density_threshold=0.5, local_neighborhood_size=0.3)
    assert matrix_ff.shape[0] == matrix_ff.shape[1] <= 36
    assert torch.allclose(matrix_ff, matrix_ff.T)
    assert torch.all(matrix_ff.diagonal() == 0)


def test_single_replicate_is_its_own_consensus():
    D_rkg, _ = planted_replicates(r=1)
    result = cc.consensus_factors(D_rkg, density_threshold=0.5, local_neighborhood_size=0.3)
    assert torch.allclose(result["consensus_D_kg"], torch.nn.functional.normalize(D_rkg[0], p=1, dim=-1))
    assert result["stability"] == 1.0
    assert result["all_neighbor_distances"] is None
    assert result["n_neighbors"] == 0


@pytest.mark.parametrize("local_neighborhood_size", [0.0, 1.0, -0.1])
def test_consensus_rejects_a_local_neighborhood_size_out_of_range(local_neighborhood_size):
    D_rkg, _ = planted_replicates()
    with pytest.raises(ValueError, match="local_neighborhood_size must be between 0 and 1"):
        cc.consensus_factors(D_rkg, density_threshold=0.5, local_neighborhood_size=local_neighborhood_size)


@pytest.mark.parametrize("density_threshold", [0.0, 1.5])
def test_consensus_rejects_a_density_threshold_out_of_range(density_threshold):
    D_rkg, _ = planted_replicates()
    with pytest.raises(ValueError, match="density_threshold must be"):
        cc.consensus_factors(D_rkg, density_threshold=density_threshold, local_neighborhood_size=0.3)


def test_consensus_rejects_a_neighborhood_of_fewer_than_two_replicates():
    D_rkg, _ = planted_replicates(r=5)  # int(5 * 0.3) = 1
    with pytest.raises(ValueError, match="too small"):
        cc.consensus_factors(D_rkg, density_threshold=0.5, local_neighborhood_size=0.3)


def test_consensus_rejects_a_density_threshold_that_leaves_too_few_replicates():
    D_rkg, _ = planted_replicates()
    with pytest.raises(ValueError, match="too few to cluster"):
        cc.consensus_factors(D_rkg, density_threshold=1e-6, local_neighborhood_size=0.3)


def test_find_local_minima_finds_the_gap_between_two_modes():
    rng = np.random.default_rng(0)
    distances = torch.from_numpy(np.concatenate([rng.normal(0.1, 0.02, 500), rng.normal(0.6, 0.02, 100)]))
    minima = cc.find_local_minima(distances.clamp(0, 1))
    # the gap is empty, so its minima are at its edges, between the modes at 0.1 and 0.6
    assert any(0.15 < m < 0.55 for m in minima)
    assert all(isinstance(m, float) for m in minima)


@pytest.mark.parametrize("search", ["fast", "exhaustive"])
def test_maximal_stability_consensus_is_at_least_as_stable_as_any_threshold_it_could_try(search):
    D_rkg, _ = planted_replicates(noise=0.4)
    limits = (0.05, 0.3)
    best = cc.maximal_stability_consensus(D_rkg, density_threshold_limits=limits, search=search)
    assert limits[0] <= best["density_threshold"] <= limits[1] + 1e-9

    stabilities = []
    for density_threshold in np.arange(limits[0], limits[1] + 1e-9, 0.01):
        try:
            stabilities.append(cc.consensus_factors(D_rkg, float(density_threshold), 0.3)["stability"])
        except ValueError:
            pass
    if search == "exhaustive":
        assert best["stability"] == pytest.approx(max(stabilities))
    else:
        assert best["stability"] <= max(stabilities) + 1e-9


def test_maximal_stability_consensus_rejects_an_unknown_search():
    D_rkg, _ = planted_replicates()
    with pytest.raises(ValueError, match="'fast' or 'exhaustive'"):
        cc.maximal_stability_consensus(D_rkg, search="slow")  # type: ignore[arg-type]


def test_maximal_stability_consensus_errors_if_no_threshold_can_cluster():
    D_rkg, _ = planted_replicates(noise=2.0)
    with pytest.raises(ValueError, match="No density threshold"):
        cc.maximal_stability_consensus(D_rkg, density_threshold_limits=(1e-6, 1e-5), search="exhaustive")
