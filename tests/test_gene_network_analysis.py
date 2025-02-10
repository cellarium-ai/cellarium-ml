# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
import pandas as pd
import pytest

from cellarium.ml.utilities.inference.gene_network_analysis import (
    GeneNetworkAnalysisBase,
    GeneralContext,
    compute_adjacency_matrix,
)

p = 50
q = 5
large_q = 20 * 4


@pytest.fixture
def z_qp() -> np.ndarray:
    """random z_qp designed to create an adjacency matrix with values in [0, 1]"""
    np.random.seed(0)
    mat = np.random.rand(q, p) * 2
    adj = mat @ mat.T
    adj = adj * (1.0 - np.eye(q))
    return mat / np.sqrt(adj.max().max())


@pytest.fixture
def structured_z_qp() -> np.ndarray:
    """random z_qp designed to create an adjacency matrix with values in [0, 1]"""
    # this has been fine tuned so that the clustering range results in different clusterings
    np.random.seed(0)
    structures = [np.random.dirichlet([10.0] * p) * 10.0 for _ in range(4)]
    mat = np.vstack([np.random.rand(large_q // 4, p) * 0.2 + structures[i] for i in range(4)])
    adj = mat @ mat.T
    adj = adj * (1.0 - np.eye(large_q))
    return mat / np.sqrt(adj.max().max())


@pytest.fixture
def ctx(structured_z_qp):
    ctx = GeneralContext(
        z_qp=structured_z_qp,
        node_names_q=[f"node_{i}" for i in range(structured_z_qp.shape[0])],
    )

    ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=10,
        self_loop=False,
        beta=3.0,
    )
    return ctx


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


def test_general_context(ctx):
    assert ctx.z_qp.shape == (large_q, p)
    assert len(ctx.node_names_q) == large_q

    print(ctx.adjacency_matrix)
    ctx.igraph()

    ctx.compute_leiden_communites(resolution=0.1)
    print(ctx.leiden_membership)

    ctx.compute_spectral_dimension()
    print(ctx.spectral_dim)


@pytest.mark.parametrize("optimization_strategy", ["gridsearch", "bayesopt"])
def test_cluster_concordance_metric(ctx, optimization_strategy):
    # a reference set made to match the structure of the data very well: we expect 4 clusters
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_q // 4)},
        "set2": {f"node_{i + large_q // 4}" for i in range(large_q // 4)},
        "set3": {f"node_{i + 2 * large_q // 4}" for i in range(large_q // 4)},
    }
    resolution_range = (0.01, 3.0)

    if optimization_strategy == "gridsearch":
        best_res, _, df, best_metrics_mean = ctx.gridsearch_optimal_resolution_communities_given_gene_sets(
            reference_gene_sets=reference_gene_sets,
            resolutions=np.linspace(*resolution_range, 20),
            metric_name="f1",
        )
    elif optimization_strategy == "bayesopt":
        best_res, _, df, best_metrics_mean = ctx.bayesopt_optimal_resolution_communities_given_gene_sets(
            reference_gene_sets=reference_gene_sets,
            resolution_range=resolution_range,
            metric_name="f1",
            num_clusterings_to_compute=20,
        )
    print(f"best_res: {best_res}")
    print(df)
    print(best_metrics_mean)

    assert best_metrics_mean > 0.9, "expected the optimal clusters to have a high f1 concordance metric"
    assert len(set(df["cluster"].unique()) - {-1}) == 4, "expected best clustering to find the 4 simulated clusters"

    # just ensure this api works
    ctx.compute_network_cluster_concordance_metric(
        reference_gene_sets=reference_gene_sets,
        resolution_range=resolution_range,
        optimization_strategy=optimization_strategy,
    )


def test_knn_concordance_metric(ctx):
    # reference_gene_sets = {
    #     "set1": {"node_0", "node_1", "node_2", "node_3"},
    #     "set2": {"node_21", "node_22", "node_23", "node_24"},
    #     "set3": {"node_50", "node_51", "node_52", "node_53"},
    # }

    # a reference set made to match the structure of the data very well: we expect 4 clusters
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_q // 4)},
        "set2": {f"node_{i + large_q // 4}" for i in range(large_q // 4)},
        "set3": {f"node_{i + 2 * large_q // 4}" for i in range(large_q // 4)},
    }
    k_values = [2, 3, 4, 10, 20, 30, 50, 75]

    best_k, _, df, best_metrics_mean = ctx.gridsearch_optimal_k_neighbors_given_gene_sets(
        reference_gene_sets=reference_gene_sets,
        k_values=k_values,
        metric_name="f1",
    )
    print(f"best_k: {best_k}")
    print(df)
    print(best_metrics_mean)

    assert best_k == 20, "expected the optimal k to be 20 which corresponds with simulated cluster sizes"
    assert best_metrics_mean > 0.5, "expected the optimal k to have a high f1 concordance metric"

    # just ensure this api works
    ctx.compute_network_knn_concordance_metric(
        reference_gene_sets=reference_gene_sets,
        k_values=k_values,
    )


def test_gene_network_analysis_base(tmpdir):
    gene_info_tsv_path = tmpdir / "gene_info.tsv"
    gene_info_df = pd.DataFrame(
        {
            "Gene Symbol": ["TTN", "RYR2", "NRXN1"],
            "ENSEMBL Gene ID": ["ENSG1", "ENSG2", "ENSG3"],
        }
    )
    gene_info_df.to_csv(gene_info_tsv_path, sep="\t", index=False)
    response_qp = np.random.randn(q, p)

    adata_obs = pd.DataFrame({"total_mrna_umis": [1000.0]})

    gene_ctx = GeneNetworkAnalysisBase(
        adata_obs=adata_obs,
        gene_info_tsv_path=gene_info_tsv_path,
        query_var_names=[f"gene_{i}" for i in range(response_qp.shape[0])],
        prompt_var_names=[f"gene_{i}" for i in range(response_qp.shape[1])],
        response_qp=response_qp,
        prompt_marginal_mean_p=np.abs(np.random.randn(p)),
        prompt_marginal_std_p=np.square(np.random.randn(p)),
        prompt_empirical_mean_p=np.abs(np.random.randn(p)),
        query_marginal_mean_q=np.abs(np.random.randn(q)),
        query_marginal_std_q=np.square(np.random.randn(q)),
        query_empirical_mean_q=np.abs(np.random.randn(q)),
    )
    original_processed_data = copy.deepcopy(gene_ctx.processed)

    gene_ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=10,
        self_loop=False,
        beta=3.0,
    )
    gene_ctx.compute_leiden_communites(resolution=0.1)
    gene_ctx.compute_spectral_dimension()

    gene_ctx.reprocess(  # more stringent cutoff values
        min_prompt_gene_tpm=25.0,
        min_query_gene_tpm=25.0,
    )
    assert gene_ctx.processed != original_processed_data

    # demonstrate we are not saving old state
    assert gene_ctx.a_qq is None
    assert gene_ctx.leiden_membership is None
    assert gene_ctx.spectral == {}

    # reprocess now
    gene_ctx.reprocess()  # default values

    assert gene_ctx.processed == original_processed_data
