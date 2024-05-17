"""Utility functions for working with gene sets"""

from gseapy.algorithm import enrichment_score, gsea_pval
import pandas as pd
import numpy as np
import io


def load_gene_set_json(file: str | bytes) -> pd.DataFrame:
    return pd.read_json(io.BytesIO(file) if isinstance(file, bytes) else file).transpose()


class GeneSetRecords:
    """
    Container to hold gene set data and provide utility functions
    """

    def __init__(self, file='~/msigdb.v2023.2.Hs.json'):
        self.df = load_gene_set_json(file)
        self._reindex()

    def __repr__(self) -> str:
        collection_names = "\n\t".join(self.get_collections())
        return f'GeneSetRecords with {len(self.df)} gene sets from the collections\n\t{collection_names}'

    def _reindex(self):

        # gene set name: set of genes
        self.gene_set_dict = self.df['geneSymbols'].to_dict()
        for k, v in self.gene_set_dict.items():
            self.gene_set_dict[k] = set(v)

        # gene name: set of gene set names
        self.gene_lookup_dict = (
            self.df['geneSymbols']
            .reset_index()
            .explode('geneSymbols')
            .groupby('geneSymbols')
            .agg(lambda s: set(s))
            .to_dict()
        )['index']

        self.msigdb_url_lookup_dict = self.df['msigdbURL'].to_dict()
        self.collections = sorted(self.df['collection'].unique().tolist())

    def get_gene_set_names(self, collection: str | None = None) -> list[str]:
        if collection is None:
            return self.df.index.tolist()
        return self.df[self.df['collection'] == collection].index.tolist()

    def get_gene_set_dict(self) -> dict[str, set[str]]:
        return self.gene_set_dict

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
                        'collection': collection,
                        'geneSymbols': gene_sets,
                    },
                    index=gene_set_names,
                ),
            ]
        )
        self._reindex()


def gsea(
    df: pd.DataFrame, 
    gene_group: list[str], 
    gene_name_key: str = 'gene_name', 
    value_key: str = 'gene_power',
    n_perm: int = 10_000,
    seed: int = 0,
):
    """
    Use GSEA to come up with a normalized enrichment score and a p-value

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
    es, es_null, hit_ind, es_g = enrichment_score(
        gene_list=df[gene_name_key],
        correl_vector=df[value_key],
        gene_set=gene_group,
        weight=1.0,
        nperm=n_perm,
        seed=seed,
    )
    pval = gsea_pval(np.array([es]), np.array([es_null])).item()
    return {'pval': pval, 'es': es}


def permutation_test(
    df: pd.DataFrame, 
    gene_group: list[str], 
    gene_name_key: str = 'gene_name', 
    value_key: str = 'gene_power',
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

    gene_group = np.unique(gene_group)

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
    collection_name: str = 'random_controls',
    sizes: list[int] = [10, 50, 100],
    repeats: int = 5,
):
    """
    Append random control gene sets to the GeneSetRecords object

    Args:
        msigdb: GeneSetRecords object
        gene_names: array of string gene names
        collection_name: name of the collection
        sizes: list of gene set sizes
        repeats: number of repeats for each size

    Returns:
        None, but appends the gene sets to the GeneSetRecords object
    """
    if len(sizes) < 1:
        raise ValueError('sizes must have at least one element')

    gene_set_names = []
    gene_sets = []
    for n_genes in sizes:
        for i in range(1, repeats + 1):
            gene_set_names.append(f'{collection_name}_{n_genes}genes_{i}')
            gene_sets.append(np.random.choice(gene_names, size=n_genes, replace=False).tolist())
    msigdb.append_collection(
        collection=collection_name,
        gene_set_names=gene_set_names,
        gene_sets=gene_sets,
    )
