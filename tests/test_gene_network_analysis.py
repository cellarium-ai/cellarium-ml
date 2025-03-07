# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
import pandas as pd
import pytest

from cellarium.ml.utilities.inference.gene_network_analysis import (
    EmpiricalCorrelationContext,
    GeneNetworkAnalysisBase,
    JacobianContext,
    NetworkAnalysisBase,
    compute_adjacency_matrix,
)

p = 5
q = 50
large_p = 20 * 4


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
    """random z_qp designed to create an adjacency matrix with 4 clusters"""
    # this has been fine tuned so that the clustering range results in different clusterings
    np.random.seed(0)
    structures = [np.random.dirichlet([1.0] * q) * 100.0 for _ in range(4)]
    mat = np.vstack([np.random.rand(large_p // 4, q) * 0.1 + structures[i] for i in range(4)])
    mat = np.abs(mat)
    # print(compute_adjacency_matrix(mat.T, beta=6.0, self_loop=False, scale_by_node_degree=True))
    # assert 0
    adj = mat @ mat.T
    adj = adj * (1.0 - np.eye(large_p))
    return mat.T / np.sqrt(adj.max().max())


@pytest.fixture
def ctx(structured_z_qp):
    ctx = NetworkAnalysisBase(
        z_qp=structured_z_qp,
        node_names_p=[f"node_{i}" for i in range(structured_z_qp.shape[1])],
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
        total_mrna_umis=None,
        query_var_names=[f"qgene_{i}" for i in range(response_qp.shape[0])],
        prompt_var_names=[f"gene_{i}" for i in range(response_qp.shape[1])],
        response_qp=response_qp,
        prompt_marginal_mean_p=np.abs(np.random.randn(response_qp.shape[1])),
        prompt_marginal_std_p=np.square(np.random.randn(response_qp.shape[1])),
        query_marginal_mean_q=np.abs(np.random.randn(response_qp.shape[0])),
        query_marginal_std_q=np.square(np.random.randn(response_qp.shape[0])),
    )
    return gene_ctx


@pytest.fixture
def jac_ctx(structured_z_qp, gene_info_tsv_path) -> JacobianContext:
    np.random.seed(0)
    response_qp = structured_z_qp  # np.random.randn(large_q, p)
    marginal_std_q = np.square(np.random.randn(response_qp.shape[0]))
    marginal_std_p = np.square(np.random.randn(response_qp.shape[1]))
    adata_obs = pd.DataFrame({"total_mrna_umis": [1000.0]})
    jac_ctx = JacobianContext(
        adata_obs=adata_obs,
        jacobian_point="test",
        gene_info_tsv_path=gene_info_tsv_path,
        query_var_names=[f"qnode_{i}" for i in range(response_qp.shape[0])],
        prompt_var_names=[f"node_{i}" for i in range(response_qp.shape[1])],
        jacobian_qp=response_qp,
        prompt_marginal_mean_p=np.ones(response_qp.shape[1]),
        prompt_marginal_std_p=marginal_std_p,
        query_marginal_mean_q=np.ones(response_qp.shape[0]),
        query_marginal_std_q=marginal_std_q,
        response_normalization_strategy="none",
        min_prompt_gene_tpm=0.0,
        min_query_gene_tpm=0.0,
        feature_max_value=10.0,
    )
    jac_ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=None,
        self_loop=False,
        beta=3.0,
        scale_by_node_degree=False,
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
    assert adjacency_matrix.shape == (p, p)
    assert np.all(adjacency_matrix >= 0.0)
    assert np.all(adjacency_matrix <= 1.0)
    # assert np.isclose(adjacency_matrix.max().max(), 1.0)
    assert np.isclose(adjacency_matrix, adjacency_matrix.T).all()
    assert np.all(adjacency_matrix.diagonal() == 0.0)


def test_network_analysis_base(ctx):
    assert ctx.z_qp.shape == (q, large_p)
    assert len(ctx.node_names_p) == large_p

    print(ctx.adjacency_matrix)
    ctx.igraph()

    ctx.compute_leiden_communites(resolution=0.1)
    print(ctx.leiden_membership)

    ctx.compute_spectral_dimension()
    print(ctx.spectral_dim)


def test_network_analysis_base_with_correlation_provided():
    ctx = NetworkAnalysisBase(
        z_qp=np.random.randn(q, q),
        z_is_correlation=True,
        node_names_p=[f"node_{i}" for i in range(q)],
    )

    assert ctx.z_is_correlation

    ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=10,
        self_loop=False,
        beta=1.0,
    )


@pytest.mark.parametrize("optimization_strategy", ["gridsearch", "bayesopt"])
def test_cluster_concordance_metric(jac_ctx, optimization_strategy):
    # a reference set made to match the structure of the data very well: we expect 4 clusters
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_p // 4)},
        "set2": {f"node_{i + large_p // 4}" for i in range(large_p // 4)},
        "set3": {f"node_{i + 2 * large_p // 4}" for i in range(large_p // 4)},
    }
    resolution_range = (0.01, 3.0)

    if optimization_strategy == "gridsearch":
        best_res, best_cl, df, best_metrics_mean = jac_ctx.gridsearch_optimal_resolution_communities_given_gene_sets(
            reference_gene_sets=reference_gene_sets,
            resolutions=np.linspace(*resolution_range, 20),
            metric_name="f1",
        )
    elif optimization_strategy == "bayesopt":
        best_res, best_cl, df, best_metrics_mean = jac_ctx.bayesopt_optimal_resolution_communities_given_gene_sets(
            reference_gene_sets=reference_gene_sets,
            resolution_range=resolution_range,
            metric_name="f1",
            num_clusterings_to_compute=20,
        )
    print(best_cl)
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
        "set1": {f"node_{i}" for i in range(large_p // 4)},
        "set2": {f"node_{i + large_p // 4}" for i in range(large_p // 4)},
        "set3": {f"node_{i + 2 * large_p // 4}" for i in range(large_p // 4)},
    }
    concordance, df = jac_ctx.compute_network_adjacency_concordance_metric(
        reference_gene_sets=reference_gene_sets,
        gene_naming="id",
    )
    print(df)
    print(concordance)
    assert ~np.isnan(concordance), "expected the adjacency concordance metric to be a valid number"


def test_compute_network_adjacency_auc_metric(jac_ctx):
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_p // 4)},
        "set2": {f"node_{i + large_p // 4}" for i in range(large_p // 4)},
        "set3": {f"node_{i + 2 * large_p // 4}" for i in range(large_p // 4)},
        "set4": {f"node_{i + 3 * large_p // 4}" for i in range(large_p // 4)},
    }
    auc, df = jac_ctx.compute_network_adjacency_auc_metric(
        reference_gene_sets=reference_gene_sets,
        gene_naming="symbol",
    )
    assert ~np.isnan(auc), "expected the adjacency concordance metric to be a valid number"
    assert df["true_positive_rate"].min() >= 0.0
    assert df["true_positive_rate"].max() <= 1.0
    assert df["false_positive_rate"].min() >= 0.0
    assert df["false_positive_rate"].max() <= 1.0
    assert auc >= 0.0
    assert auc <= 1.0

    assert auc > 0.95, "expected the adjacency auc metric to be high"

    # run it again
    auc2, df = jac_ctx.compute_network_adjacency_auc_metric(
        reference_gene_sets=reference_gene_sets,
        gene_naming="symbol",
    )
    assert auc2 == auc, "non-deterministic behavior in adjacency auc metric"


def test_compute_network_adjacency_auc_metric_per_gene(jac_ctx):
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_p // 4)},
        "set2": {f"node_{i + large_p // 4}" for i in range(large_p // 4)},
        "set3": {f"node_{i + 2 * large_p // 4}" for i in range(large_p // 4)},
        "set4": {f"node_{i + 3 * large_p // 4}" for i in range(large_p // 4)},
    }
    auc_p, tpr_pp, fpr_pp = jac_ctx.compute_network_adjacency_auc_metric_per_gene(
        reference_gene_sets=reference_gene_sets,
        gene_naming="symbol",
    )
    print("computed AUCs per gene:")
    print(auc_p)
    assert (~np.isnan(auc_p)).any(), "expected the adjacency concordance metrics to be valid numbers"
    assert tpr_pp.min() >= 0.0
    assert tpr_pp.max() <= 1.0
    assert fpr_pp.min() >= 0.0
    assert fpr_pp.max() <= 1.0
    assert (auc_p >= 0.0).all()
    assert (auc_p <= 1.0).all()

    assert auc_p.mean() > 0.95, "expected the mean per-gene adjacency auc metric to be high"

    # run it again
    auc2_p, _, _ = jac_ctx.compute_network_adjacency_auc_metric_per_gene(
        reference_gene_sets=reference_gene_sets,
        gene_naming="symbol",
    )
    assert (auc2_p == auc_p).all(), "non-deterministic behavior in per-gene adjacency auc metric"


@pytest.mark.parametrize("metric_name", ["f1", "precision"])
def test_knn_concordance_metric(jac_ctx, metric_name):
    # a reference set made to match the structure of the data very well: we expect 4 clusters
    reference_gene_sets = {
        "set1": {f"node_{i}" for i in range(large_p // 4)},
        "set2": {f"node_{i + large_p // 4}" for i in range(large_p // 4)},
        "set3": {f"node_{i + 2 * large_p // 4}" for i in range(large_p // 4)},
        "set4": {f"node_{i + 3 * large_p // 4}" for i in range(large_p // 4)},
    }
    k_values = [2, 3, 4, 10, 20, 30, 50, 75]

    best_metrics_df = jac_ctx.gridsearch_optimal_k_neighbors_given_gene_sets(
        reference_gene_sets=reference_gene_sets,
        k_values=k_values,
        metric_name=metric_name,
    )
    best_k = best_metrics_df["k"].value_counts().index[0]
    print(best_metrics_df["k"].value_counts())
    print(f"best_k: {best_k}")
    print(best_metrics_df)
    best_metrics_mean = best_metrics_df[metric_name].mean()
    print(best_metrics_mean)

    if metric_name == "f1":
        assert best_k == 20, "expected the optimal k to be 20 which corresponds with simulated cluster sizes"
    assert best_metrics_mean > 0.9, "expected the optimal k to have a high f1 concordance metric"

    # just ensure this api works
    jac_ctx.compute_network_knn_concordance_metric(
        reference_gene_sets=reference_gene_sets,
        k_values=k_values,
        metric_name=metric_name,
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
    assert gene_ctx.a_pp is None
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


def test_empirical_correlation_context(gene_info_tsv_path):
    # just show the api works without error
    ctx = EmpiricalCorrelationContext(
        gene_info_tsv_path=gene_info_tsv_path,
        total_mrna_umis=None,
        var_names_g=[f"gene_{i}" for i in range(20)],
        covariance_gg=np.random.rand(20, 20),
        marginal_mean_g=np.abs(np.random.randn(20)),
        marginal_std_g=np.square(np.random.randn(20)),
    )
    print(ctx.z_qp)
    ctx.compute_adjacency_matrix(
        adjacency_strategy="positive_correlation",
        n_neighbors=None,
        self_loop=False,
        beta=1.0,
    )
    ctx.igraph()
    ctx.compute_leiden_communites(resolution=0.1)
    ctx.compute_spectral_dimension()
    ctx.reprocess()


# def test_validation_mixin():
#     jac_ctx
