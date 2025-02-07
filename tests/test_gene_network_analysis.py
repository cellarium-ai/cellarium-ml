# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from cellarium.ml.utilities.inference.gene_network_analysis import (
    GeneralContext,
    compute_adjacency_matrix,
)

p = 50
q = 5
large_q = 20 * 4


@pytest.fixture
def z_qp() -> np.ndarray:
    """random z_qp designed to create an adjacency matrix with values in [0, 1]"""
    mat = np.random.rand(q, p) * 2
    adj = mat @ mat.T
    adj = adj * (1.0 - np.eye(q))
    return mat / np.sqrt(adj.max().max())


@pytest.fixture
def structured_z_qp() -> np.ndarray:
    """random z_qp designed to create an adjacency matrix with values in [0, 1]"""
    # this has been fine tuned so that the clustering range results in different clusterings
    structures = [np.random.dirichlet([10.0] * p) * 10.0 for _ in range(4)]
    mat = np.vstack([np.random.rand(large_q // 4, p) * 0.2 + structures[i] for i in range(4)])
    adj = mat @ mat.T
    adj = adj * (1.0 - np.eye(large_q))
    return mat / np.sqrt(adj.max().max())


@pytest.fixture
def ctx(structured_z_qp) -> GeneralContext:
    return GeneralContext(
        z_qp=structured_z_qp,
        node_names_q=[f"node_{i}" for i in range(structured_z_qp.shape[0])],
    )


@pytest.mark.parametrize("n_neighbors", [None, 2, 3])
def test_compute_adjacency_matrix(z_qp, n_neighbors):
    adjacency_matrix = compute_adjacency_matrix(
        z_qp=z_qp,
        adjacency_strategy="positive_correlation",
        n_neighbors=n_neighbors,
        self_loop=False,
        beta=2.0,
    )
    print(adjacency_matrix)
    assert adjacency_matrix.shape == (q, q)
    assert np.all(adjacency_matrix >= 0.0)
    assert np.isclose(adjacency_matrix.max().max(), 1.0)
    assert np.isclose(adjacency_matrix, adjacency_matrix.T).all()
    assert np.all(adjacency_matrix.diagonal() == 0.0)


@pytest.mark.parametrize("optimization_strategy", ["gridsearch", "bayesopt"])
def test_general_context(ctx, optimization_strategy):
    assert ctx.z_qp.shape == (large_q, p)
    assert len(ctx.node_names_q) == large_q

    ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=10,
        self_loop=False,
        beta=3.0,
    )
    print(ctx.adjacency_matrix)
    ctx.igraph()

    ctx.compute_leiden_communites(resolution=0.1)
    print(ctx.leiden_membership)

    ctx.compute_spectral_dimension()
    print(ctx.spectral_dim)

    ctx.compute_network_concordance_metric(
        reference_gene_sets={
            "set1": {"node_0", "node_1", "node_2", "node_3"},
            "set2": {"node_21", "node_22"},
            "set3": {"node_50", "node_51", "node_52", "node_53"},
        },
        resolution_range=(0.01, 3.0),
        optimization_strategy=optimization_strategy,
    )
