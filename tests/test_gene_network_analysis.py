# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
import pandas as pd
import pytest

from cellarium.ml.utilities.inference.gene_network_analysis import (
    GeneNetworkAnalysisBase,
    GeneralContext,
    JacobianContext,
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


@pytest.fixture
def gene_info_tsv_path(tmpdir) -> str:
    gene_info_tsv_path = tmpdir / "gene_info.tsv"
    gene_info_df = pd.DataFrame(
        {
            "Gene Symbol": ["TTN", "RYR2", "NRXN1"],
            "ENSEMBL Gene ID": ["ENSG1", "ENSG2", "ENSG3"],
        }
    )
    gene_info_df.to_csv(gene_info_tsv_path, sep="\t", index=False)
    return gene_info_tsv_path


@pytest.fixture
def gene_ctx(structured_z_qp, gene_info_tsv_path) -> GeneNetworkAnalysisBase:
    response_qp = structured_z_qp  # np.random.randn(large_q, p)
    adata_obs = pd.DataFrame({"total_mrna_umis": [1000.0]})
    gene_ctx = GeneNetworkAnalysisBase(
        adata_obs=adata_obs,
        gene_info_tsv_path=gene_info_tsv_path,
        query_var_names=[f"gene_{i}" for i in range(response_qp.shape[0])],
        prompt_var_names=[f"gene_{i}" for i in range(response_qp.shape[1])],
        response_qp=response_qp,
        prompt_marginal_mean_p=np.abs(np.random.randn(response_qp.shape[1])),
        prompt_marginal_std_p=np.square(np.random.randn(response_qp.shape[1])),
        prompt_empirical_mean_p=np.abs(np.random.randn(response_qp.shape[1])),
        query_marginal_mean_q=np.abs(np.random.randn(response_qp.shape[0])),
        query_marginal_std_q=np.square(np.random.randn(response_qp.shape[0])),
        query_empirical_mean_q=np.abs(np.random.randn(response_qp.shape[0])),
    )
    return gene_ctx


@pytest.fixture
def jac_ctx(structured_z_qp, gene_info_tsv_path) -> JacobianContext:
    response_qp = structured_z_qp  # np.random.randn(large_q, p)
    adata_obs = pd.DataFrame({"total_mrna_umis": [1000.0]})
    jac_ctx = JacobianContext(
        adata_obs=adata_obs,
        jacobian_point="test",
        gene_info_tsv_path=gene_info_tsv_path,
        query_var_names=[f"node_{i}" for i in range(response_qp.shape[0])],
        prompt_var_names=[f"node_{i}" for i in range(response_qp.shape[1])],
        jacobian_qp=response_qp,
        prompt_marginal_mean_p=np.abs(np.random.randn(response_qp.shape[1])),
        prompt_marginal_std_p=np.square(np.random.randn(response_qp.shape[1])),
        prompt_empirical_mean_p=np.abs(np.random.randn(response_qp.shape[1])),
        query_marginal_mean_q=np.abs(np.random.randn(response_qp.shape[0])),
        query_marginal_std_q=np.square(np.random.randn(response_qp.shape[0])),
        query_empirical_mean_q=np.abs(np.random.randn(response_qp.shape[0])),
    )
    jac_ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=10,
        self_loop=False,
        beta=3.0,
    )
    return jac_ctx


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
def test_cluster_concordance_metric(jac_ctx, optimization_strategy):
    # a reference set made to match the structure of the data very well: we expect 4 clusters
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_q // 4)},
        "set2": {f"node_{i + large_q // 4}" for i in range(large_q // 4)},
        "set3": {f"node_{i + 2 * large_q // 4}" for i in range(large_q // 4)},
    }
    resolution_range = (0.01, 3.0)

    if optimization_strategy == "gridsearch":
        best_res, _, df, best_metrics_mean = jac_ctx.gridsearch_optimal_resolution_communities_given_gene_sets(
            reference_gene_sets=reference_gene_sets,
            resolutions=np.linspace(*resolution_range, 20),
            metric_name="f1",
        )
    elif optimization_strategy == "bayesopt":
        best_res, _, df, best_metrics_mean = jac_ctx.bayesopt_optimal_resolution_communities_given_gene_sets(
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
    jac_ctx.compute_network_cluster_concordance_metric(
        reference_gene_sets=reference_gene_sets,
        resolution_range=resolution_range,
        optimization_strategy=optimization_strategy,
    )


def test_compute_network_adjacency_concordance_metric(jac_ctx):
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_q // 4)},
        "set2": {f"node_{i + large_q // 4}" for i in range(large_q // 4)},
        "set3": {f"node_{i + 2 * large_q // 4}" for i in range(large_q // 4)},
    }
    concordance, df = jac_ctx.compute_network_adjacency_concordance_metric(
        reference_gene_sets=reference_gene_sets,
        gene_naming="id",
    )
    print(df)
    print(concordance)
    assert ~np.isnan(concordance), "expected the adjacency concordance metric to be a valid number"


def test_knn_concordance_metric(jac_ctx):
    # a reference set made to match the structure of the data very well: we expect 4 clusters
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_q // 4)},
        "set2": {f"node_{i + large_q // 4}" for i in range(large_q // 4)},
        "set3": {f"node_{i + 2 * large_q // 4}" for i in range(large_q // 4)},
    }
    k_values = [2, 3, 4, 10, 20, 30, 50, 75]

    best_k, _, df, best_metrics_mean = jac_ctx.gridsearch_optimal_k_neighbors_given_gene_sets(
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
    jac_ctx.compute_network_knn_concordance_metric(
        reference_gene_sets=reference_gene_sets,
        k_values=k_values,
    )


def test_gene_network_analysis_base(gene_ctx):
    original_processed_data = copy.deepcopy(gene_ctx.processed)

    gene_ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=10,
        self_loop=False,
        beta=3.0,
    )
    gene_ctx.compute_spectral_dimension()
    gene_ctx.compute_leiden_communites(resolution=0.1)
    assert gene_ctx.adjacency_matrix.shape[0] == gene_ctx.leiden_communities.shape[0]

    gene_ctx.reprocess(  # more stringent cutoff values
        min_prompt_gene_tpm=500.0,
        min_query_gene_tpm=750.0,
    )

    # demonstrate we are not saving old state
    assert gene_ctx.a_qq is None
    assert gene_ctx.leiden_membership is None
    assert gene_ctx.spectral == {}

    # ensure the leiden and adjacency dimensions agree
    gene_ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=20,
        self_loop=False,
        beta=4.0,
    )
    assert gene_ctx.processed != original_processed_data
    first_igraph = gene_ctx.igraph()
    res = 2.0
    first_leiden = gene_ctx.compute_leiden_communites(resolution=res)

    assert gene_ctx.adjacency_matrix.shape[0] == gene_ctx.leiden_communities.shape[0]

    # recompute adjacency matrix with different parameters
    gene_ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=3,
        self_loop=False,
        beta=1.0,
    )
    second_igraph = gene_ctx.igraph()
    second_leiden = gene_ctx.compute_leiden_communites(resolution=res)

    # ensure graph didn't cache the old results
    assert first_igraph != second_igraph, "expected different igraph objects with different adjacency matrix parameters"

    # ensure leiden didn't cache the old results
    assert (first_leiden.shape != second_leiden.shape) or (first_leiden != second_leiden).any(), (
        "expected different clusterings with different adjacency matrix parameters"
    )

    # reprocess now
    gene_ctx.reprocess()  # default values

    assert gene_ctx.processed == original_processed_data

    assert max(gene_ctx.query_gene_id_to_idx_map.values()) < gene_ctx.z_qp.shape[0]
    assert max(gene_ctx.query_gene_id_to_idx_map.values()) < len(gene_ctx.processed.query_var_names)


# def test_validation_mixin():
#     jac_ctx
