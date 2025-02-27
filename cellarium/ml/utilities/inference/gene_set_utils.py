# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for working with gene sets."""

import io
from typing import Literal

import numpy as np
import pandas as pd
from gseapy.algorithm import enrichment_score, gsea_pval


def load_gene_set_json(file: str | bytes) -> pd.DataFrame:
    return pd.read_json(io.BytesIO(file) if isinstance(file, bytes) else file).transpose()


class GeneSetRecords:
    """
    Container to hold gene set data and provide utility functions
    """

    def __init__(self, file: str):
        # file obtained from https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2024.1.Hs/msigdb.v2024.1.Hs.json
        self.df = load_gene_set_json(file)
        self._reindex()

    def __repr__(self) -> str:
        collection_names = "\n\t".join(self.get_collections())
        return f"GeneSetRecords with {len(self.df)} gene sets from the collections\n\t{collection_names}"

    def _reindex(self):
        # gene set name: set of genes
        self.gene_set_dict = self.df["geneSymbols"].to_dict()
        for k, v in self.gene_set_dict.items():
            self.gene_set_dict[k] = set(v)

        # gene name: set of gene set names
        self.gene_lookup_dict = (
            self.df["geneSymbols"]
            .reset_index()
            .explode("geneSymbols")
            .groupby("geneSymbols")
            .agg(lambda s: set(s))
            .to_dict()
        )["index"]

        self.msigdb_url_lookup_dict = self.df["msigdbURL"].to_dict()
        self.collections = sorted(self.df["collection"].unique().tolist())

    def get_gene_set_names(self, collection: str | None = None) -> list[str]:
        if collection is None:
            return self.df.index.tolist()
        return self.df[self.df["collection"] == collection].index.tolist()

    def get_gene_set_dict(self) -> dict[str, set[str]]:
        return self.gene_set_dict

    def get_gene_set(self, gene_set_name: str) -> set[str]:
        return self.gene_set_dict[gene_set_name]

    def get_gene_set_dict_for_collection(self, collection: str) -> dict[str, set[str]]:
        collection_gene_set_dict = self.df["geneSymbols"][self.df["collection"] == collection].to_dict()
        for k, v in collection_gene_set_dict.items():
            collection_gene_set_dict[k] = set(v)
        return collection_gene_set_dict

    def get_gene_lookup_dict(self) -> dict[str, set[str]]:
        return self.gene_lookup_dict

    def get_collections(self) -> list[str]:
        return self.collections

    def get_msigdb_url(self, gene_set_name: str) -> str | None:
        return self.msigdb_url_lookup_dict.get(gene_set_name, None)

    def append_collection(self, collection: str, gene_set_names: list[str], gene_sets: list[list[str]]) -> None:
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    {
                        "collection": collection,
                        "geneSymbols": gene_sets,
                    },
                    index=gene_set_names,
                ),
            ]
        )
        self._reindex()

    def append_random_control_collection(
        self,
        gene_names: np.ndarray,
        collection_name: str,
        sizes: list[int],
        repeats: int,
    ) -> None:
        append_random_control_collection(
            msigdb=self,
            gene_names=gene_names,
            collection_name=collection_name,
            sizes=sizes,
            repeats=repeats,
        )

    def remove_gene_sets(self, gene_set_names: list[str]) -> None:
        self.df = self.df.drop(gene_set_names)
        self._reindex()

    def remove_gene_sets_from_collection(self, gene_set_names: list[str], collection: str) -> None:
        collection_df = self.df[self.df["collection"] == collection]
        collection_df = collection_df.drop(gene_set_names)
        self._reindex()


def gsea(
    df: pd.DataFrame,
    gene_group: list[str],
    gene_name_key: str = "gene_name",
    value_key: str = "gene_power",
    n_perm: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Use GSEApy to come up with an enrichment score and a p-value

    Args:
        df: DataFrame with gene names and values
        gene_group: list of gene names
        gene_name_key: column name for gene names
        value_key: column name for values
        n_perm: number of permutations
        seed: random seed

    Returns:
        dict with 'pval' p-value, 'es' enrichment score
    """
    # check for a failure condition: all the genes in the group have a zero loading
    n_genes_with_nonzero_loading = sum([g in gene_group for g in df[df[value_key] != 0][gene_name_key].unique()])
    if n_genes_with_nonzero_loading == 0:
        return {"pval": 1.0, "es": np.nan}

    es, es_null, hit_ind, es_g = enrichment_score(
        gene_list=df[gene_name_key].unique(),
        correl_vector=df[value_key],
        gene_set=np.unique(gene_group),
        weight=1.0,
        nperm=n_perm,
        seed=seed,
    )
    pval = gsea_pval(np.array([es]), np.array([es_null])).item()
    return {"pval": pval, "es": es}


def permutation_test(
    df: pd.DataFrame,
    gene_group: list[str],
    gene_name_key: str = "gene_name",
    value_key: str = "gene_power",
    n_perm: int = 10_000,
    seed: int = 0,
):
    """
    Perform a permutation test to compute the p-value of the mean of a gene group

    Args:
        df: DataFrame with gene names and values
        gene_group: list of gene names
        gene_name_key: column name for gene names
        value_key: column name for values
        n_perm: number of permutations
        seed: random seed

    Returns:
        gene group mean, mean of the permuted values, p-value
    """
    np.random.seed(seed)

    gene_group = np.unique(gene_group).tolist()  # unique elements

    means = []

    for i in range(n_perm):
        # pick a set of same size
        values = np.random.choice(df[value_key].values, size=len(gene_group), replace=False)

        # compute mean
        means.append(np.mean(values).item())

    # compute the mean of the gene group
    gene_group_mean = df[df[gene_name_key].isin(gene_group)][value_key].mean()

    # compute the p-value
    p_value = np.mean(np.array(means) > gene_group_mean)

    return gene_group_mean, np.mean(np.array(means)), p_value


def append_random_control_collection(
    msigdb: GeneSetRecords,
    gene_names: np.ndarray,
    collection_name: str = "random_controls",
    sizes: list[int] = [10, 50, 100],
    repeats: int = 5,
):
    """
    Append random control gene sets to the :class:`GeneSetRecords` object.
    The idea is to use this as a control to compare against the real gene sets,
    sort of like a permutation test that would calibrate computed metrics.

    Args:
        msigdb: :class:`GeneSetRecords` object to append random gene sets to
        gene_names: array of string gene names
        collection_name: name of the collection
        sizes: list of gene set sizes
        repeats: number of repeats for each size

    Returns:
        None, but appends the gene sets to the :class:`GeneSetRecords` object
    """
    if len(sizes) < 1:
        raise ValueError("sizes must have at least one element")

    gene_set_names = []
    gene_sets = []
    for n_genes in sizes:
        for i in range(1, repeats + 1):
            gene_set_names.append(f"{collection_name}_{n_genes}genes_{i}")
            gene_sets.append(np.random.choice(gene_names, size=n_genes, replace=False).tolist())
    msigdb.append_collection(
        collection=collection_name,
        gene_set_names=gene_set_names,
        gene_sets=gene_sets,
    )


def compute_function_on_gene_sets(
    input_gene_set: set[str],
    reference_gene_sets: dict[str, set[str]],
    metric_name: Literal["iou", "intersection", "precision", "precision_recall", "f1"],
) -> pd.Series:
    """
    Given a set of genes and a dictionary of gene sets, compute a function measuring
    something about the overlap between the input set and each predefined gene set.

    Args:
        input_gene_set: set of gene names
        reference_gene_sets: dictionary of gene set names to set of gene names
            (e.g. from gseapy.get_library(name=...) or :meth:`GeneSetRecords.get_gene_set_dict`)
        metric_name: one of
            - 'iou': intersection over union
            - 'intersection': number of genes in input and reference gene set
            - 'precision': intersection divided by number of genes in input set
            - 'precision_recall': precision and recall (intersection divided by
                number of genes in reference set)
            - 'f1': f1 score (harmonic mean of precision and recall)

    Returns:
        pd.Series with the function value for each gene set
    """
    results = {}
    for name, gset in reference_gene_sets.items():
        intersection = len(input_gene_set.intersection(gset))
        match metric_name:
            case "iou":
                union = len(input_gene_set.union(gset))  # few extra ms
                metric: float | tuple[float, float] = intersection / union
            case "intersection":
                metric = intersection
            case "precision":
                metric = intersection / len(input_gene_set)
            case "precision_recall":
                precision = intersection / len(input_gene_set)
                recall = intersection / len(gset)
                metric = (precision, recall)
            case "f1":
                # precision = intersection / len(in_gset)
                # recall = intersection / len(gset)
                # metric = 2 * (precision * recall) / (precision + recall + 1e-10)
                metric = 2 * intersection / (len(input_gene_set) + len(gset))  # same as above
            case _:
                raise ValueError(f"Unknown function {metric_name}")
        results[name] = metric
    return pd.Series(results, name=metric_name)


def compute_function_on_gene_sets_given_clustering(
    clustering: np.ndarray,
    gene_names: np.ndarray,
    reference_gene_sets: dict[str, set[str]],
    metric_name: Literal["iou", "intersection", "precision", "precision_recall", "f1"],
) -> pd.DataFrame:
    """
    Given gene cluster labels and a dictionary of reference gene sets, compute the best matching
    reference gene set for each cluster and record the corresponding metric.

    Args:
        clustering: array of cluster labels
        gene_names: array of gene names, same order as clustering
        reference_gene_sets: dictionary of gene set names to set of gene names
        metric_name: one of ['iou', 'intersection', 'precision', 'precision_recall', 'f1'].
            see :func:`compute_function_on_gene_sets`

    Returns:
        DataFrame with columns ['cluster', 'reference_gene_set', metric_name]
    """
    all_genes_in_reference = set.union(*reference_gene_sets.values())
    if len(set(gene_names).intersection(all_genes_in_reference)) == 0:
        raise ValueError("No overlap between genes in gene_names and reference_gene_sets")
    assert len(clustering) == len(gene_names), "clustering and gene_names must have the same length"

    # for each cluster, compute the best matching reference gene set and record its metric
    best_metric_dfs: list[pd.DataFrame] = []
    for k in np.unique(clustering):
        # # skip the noise cluster
        # if k == -1:
        #     continue

        # genes in this cluster
        gene_set = set(gene_names[clustering == k])

        # compute the metric for each reference gene set
        metrics = compute_function_on_gene_sets(
            input_gene_set=gene_set,
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
        )

        # record the best metric and the corresponding reference gene set name
        best_metric_dfs.append(
            pd.DataFrame({"cluster": [k], "reference_gene_set": [metrics.idxmax()], metric_name: [metrics.max()]})
        )
    best_metric_df = (
        pd.concat(best_metric_dfs, axis=0)
        if best_metric_dfs
        else pd.DataFrame(data={"cluster": [-1], "reference_gene_set": [np.nan], metric_name: [0.0]})
    )

    return best_metric_df


def compute_function_on_gene_sets_given_neighbors(
    neighbor_lookup: dict[str, set[str]],
    reference_gene_sets: dict[str, set[str]],
    metric_name: Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "iou",
) -> pd.DataFrame:
    """
    For each gene in neighbor_lookup.keys(), compute metrics over all the gene sets.
    Calls :func:`compute_function_on_gene_sets`.

    Example:
        >>> import gseapy as gp
        >>> from cellarium.ml.downstream.noise_prompting import compute_knn_dict
        >>> gene_sets = gp.get_library('Reactome_2022')
        >>> neighbor_lookup = compute_knn_dict(adata_tpm, k=3, obs_gene_key='perturbation')
        >>> best_gene_sets = compute_top_gene_set_per_gene(neighbor_lookup, gene_sets, func='iou')

    Args:
        neighbor_lookup: dictionary of gene names to set of gene names
        reference_gene_sets: dictionary of gene set names to set of gene names
        metric_name: one of ['iou', 'intersection', 'precision', 'precision_recall', 'f1'].
            see :func:`compute_function_on_gene_sets`

    Returns:
        DataFrame with columns ['gene', 'gene_set', metric_name] which includes the best gene set
        for each gene and the corresponding metric value
    """
    all_genes_in_knn = set.union(*neighbor_lookup.values())
    all_genes_in_reference = set.union(*reference_gene_sets.values())
    if len(all_genes_in_knn.intersection(all_genes_in_reference)) == 0:
        raise ValueError("No overlap between genes in neighbor_lookup and reference_gene_sets")

    best_gene_set_dfs: list[pd.DataFrame] = []

    from tqdm import tqdm

    for gene in tqdm(neighbor_lookup.keys()):
        metric_series = compute_function_on_gene_sets(
            input_gene_set=neighbor_lookup[gene],
            reference_gene_sets=reference_gene_sets,
            metric_name=metric_name,
        )
        best_gene_set_dfs.append(
            pd.DataFrame({"gene": [gene], "gene_set": [metric_series.idxmax()], metric_name: [metric_series.max()]})
        )

    best_gene_set_df = (
        pd.concat(best_gene_set_dfs, axis=0)
        if best_gene_set_dfs
        else pd.DataFrame(data={"gene": [np.nan], "gene_set": [np.nan], metric_name: [0.0]})
    )
    return best_gene_set_df


def compute_top_gene_set_per_gene(
    neighbor_lookup: dict[str, set[str]],
    reference_gene_sets: dict[str, set[str]],
    metric_name: Literal["iou", "intersection", "precision", "precision_recall", "f1"] = "iou",
) -> dict[str, str]:
    """
    For each gene in neighbor_lookup.keys(), compute the gene set with the highest metric
    over all the gene sets. Calls :func:`compute_function_on_gene_sets_given_neighbors`.
    Useful for finding a gene set label for plotting, without needing to cluster the data.

    Example:
        import gseapy as gp
        from cellarium.ml.downstream.noise_prompting import compute_knn_dict
        reference_gene_sets = gp.get_library('Reactome_2022')
        neighbor_lookup = compute_knn_dict(adata_tpm, k=3, obs_gene_key='perturbation')
        best_gene_sets = compute_top_gene_set_per_gene(neighbor_lookup, reference_gene_sets, func='iou')

    Args:
        neighbor_lookup: dictionary of gene names to set of gene names
        reference_gene_sets: dictionary of gene set names to set of gene names
        metric_name: one of ['iou', 'intersection', 'precision', 'precision_recall', 'f1'].
            see :func:`compute_function_on_gene_sets`

    Returns:
        dictionary of gene names to top gene set names
    """
    df = compute_function_on_gene_sets_given_neighbors(
        neighbor_lookup=neighbor_lookup,
        reference_gene_sets=reference_gene_sets,
        metric_name=metric_name,
    )
    return df.set_index("gene")["gene_set"].to_dict()
