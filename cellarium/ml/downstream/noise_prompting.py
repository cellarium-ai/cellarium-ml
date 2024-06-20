"""Noise prompting using the CellariumGPT model"""

from cellarium.ml.core import CellariumPipeline
from cellarium.ml.models.cellarium_gpt import NegativeBinomial

import torch
import numpy as np
import scipy.sparse as sp
import anndata
import scanpy as sc
from scanpy._utils import axis_mul_or_truediv
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.metrics import explained_variance_score
from tqdm import tqdm
import io
import os
from operator import truediv
import warnings
from contextlib import redirect_stdout
import tempfile
import asyncio

from .cellarium_utils import get_datamodule
from .gene_set_utils import GeneSetRecords, append_random_control_collection, gsea


@torch.no_grad()
def gpt_predict(
    adata: anndata.AnnData, 
    pipeline: CellariumPipeline, 
    gene_inds: torch.LongTensor,
    layer: str,
    key_added: str,
) -> anndata.AnnData:
    """
    Use the pipeline to predict the expression of a subset of genes in a single cell.

    Args:
        adata: AnnData object with a single cell
        pipeline: CellariumPipeline object
        gene_inds: indices of genes to predict
        layer: layer in adata.layers to use
        key_added: key to add to adata.layers

    Returns:
        AnnData object with predicted expression in adata.layers[key_added]
    """
    adata.X = adata.layers[layer].copy()
    dm = get_datamodule(adata)

    device = pipeline[-1].gpt_model.parameters().__next__().device
    gene_inds = torch.unique(gene_inds).sort().values.long().to(device)
    adata_out = adata[:, gene_inds.cpu().numpy()].copy()

    # predict
    i = 0
    for batch in dm.predict_dataloader():
        batch["x_ng"] = batch["x_ng"].to(device)
        batch["total_mrna_umis_n"] = batch["x_ng"].sum(-1)
        batch["context_inds_c"] = gene_inds
        out = pipeline.predict(batch)
        adata_out.X[i:i + batch["x_ng"].shape[0]] = out["mu_nc"][:, 1:].cpu().numpy()
        i += batch["x_ng"].shape[0]

    adata_out.layers[key_added] = adata_out.X.copy()
    
    return adata_out


# @torch.no_grad()
# def create_randomly_perturbed_dataset_from_manifold(
#     adata, 
#     n_perturbations: int, 
#     scale_per_gene_g: np.ndarray,
#     manifold_mu_obsm_key: str = 'measured_gpt',
#     manifold_theta_obsm_key: str = 'measured_gpt_disp',
#     perturbation_scale: float = 0.1,
#     square_epsilon: bool = False,
#     genes_to_perturb: list[int] | np.ndarray = [],
#     seed: int = 0,
# ):
#     """
#     Create a dataset with perturbed expression.

#     Args:
#         adata: AnnData object with a single cell
#         n_perturbations: number of perturbations to create
#         scale_per_gene_g: scale for each gene
#         perturbation_scale: scale of the perturbation
#         square_epsilon: whether to square the perturbation
#         genes_to_perturb: indices of genes to perturb
#         cell_perturbations_coherent: whether to perturb all genes in a cell in the same direction
#             and by the same amount
#         seed: random seed

#     Returns:
#         AnnData object of size n_perturbations with perturbed expression in adata.layers["perturbed"]
#     """
    
#     assert len(adata) == 1, "Only one cell allowed in adata for create_perturbed_dataset"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     adata_perturbed = anndata.concat([adata for _ in range(n_perturbations)], 
#                                      axis=0, merge="same", index_unique="-pert", uns_merge="first")
#     adata_perturbed.layers["measured"] = adata_perturbed.X.copy()
#     if sp.issparse(adata_perturbed.X):
#         counts_ng = adata_perturbed.X.toarray()
#     else:
#         counts_ng = adata_perturbed.X
#     counts_ng = torch.from_numpy(counts_ng).to(device)
#     torch.manual_seed(seed)

#     # re-sample counts
#     NegativeBinomial(mu=adata.obsm[manifold_mu_obsm_key], theta=adata.obsm[manifold_theta_obsm_key]).sample()

#     # restrict the perturbation to genes specified


#     adata_perturbed.X = sp.csr_matrix(perturbed_counts_ng.cpu().numpy())
#     assert adata_perturbed.X.min() >= 0, "Negative counts in the perturbed dataset"
#     adata_perturbed.layers["perturbed"] = adata_perturbed.X.copy()

#     return adata_perturbed


@torch.no_grad()
def create_perturbed_dataset(
    adata, 
    n_perturbations: int, 
    scale_per_gene_g: np.ndarray,
    perturbation_scale: float = 0.1,
    square_epsilon: bool = False,
    genes_to_perturb: list[int] | np.ndarray = [],
    cell_perturbations_coherent: bool = False,
    seed: int = 0,
):
    """
    Create a dataset with perturbed expression.

    Args:
        adata: AnnData object with a single cell
        n_perturbations: number of perturbations to create
        scale_per_gene_g: scale for each gene
        perturbation_scale: scale of the perturbation
        square_epsilon: whether to square the perturbation
        genes_to_perturb: indices of genes to perturb
        cell_perturbations_coherent: whether to perturb all genes in a cell in the same direction
            and by the same amount
        seed: random seed

    Returns:
        AnnData object of size n_perturbations with perturbed expression in adata.layers["perturbed"]
    """
    
    assert len(adata) == 1, "Only one cell allowed in adata for create_perturbed_dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata_perturbed = anndata.concat([adata for _ in range(n_perturbations)], 
                                     axis=0, merge="same", index_unique="-pert", uns_merge="first")
    adata_perturbed.layers["measured"] = adata_perturbed.X.copy()
    if sp.issparse(adata_perturbed.X):
        counts_ng = adata_perturbed.X.toarray()
    else:
        counts_ng = adata_perturbed.X
    counts_ng = torch.from_numpy(counts_ng).to(device)
    torch.manual_seed(seed)

    if cell_perturbations_coherent:
        # all genes in a cell go the same way, by the same amount, i.e. epsilon_ng = epsilon_n
        epsilon_ng = torch.randn(counts_ng.shape[0], 1, device=device).expand(-1, counts_ng.shape[1])
    else:
        epsilon_ng = torch.randn(counts_ng.shape, device=device)

    epsilon_ng_sign = epsilon_ng.sign()
    
    if square_epsilon:
        epsilon_ng = epsilon_ng.square()

    scale_per_gene_g = torch.from_numpy(scale_per_gene_g).to(device).squeeze()

    # define the perturbation
    assert (scale_per_gene_g > 0).all(), "automatically-compute gene scales are negative: do you have raw counts in adata.X ?"
    perturbation_ng = epsilon_ng_sign * torch.distributions.Poisson(
        perturbation_scale * (epsilon_ng.abs() + 1e-10) * scale_per_gene_g.unsqueeze(0)
    ).sample()

    # restrict the perturbation to genes specified
    if len(genes_to_perturb) > 0:
        genes_to_perturb = torch.tensor(genes_to_perturb, device=device).long()
        unperturbed_genes_g = torch.zeros(counts_ng.shape[1], device=device).scatter_(0, genes_to_perturb, 1).bool().logical_not()
        perturbation_ng[:, unperturbed_genes_g] = 0.0

    # apply the perturbation
    perturbed_counts_ng = torch.clamp(counts_ng + perturbation_ng, min=0.0).squeeze()

    adata_perturbed.X = sp.csr_matrix(perturbed_counts_ng.cpu().numpy())
    assert adata_perturbed.X.min() >= 0, "Negative counts in the perturbed dataset"
    adata_perturbed.layers["perturbed"] = adata_perturbed.X.copy()

    return adata_perturbed


def get_gene_integer_indices(adata, genes: list[str], gene_name_key: str = 'gene_name') -> list[int]:
    """
    Get integer indices of a list of genes in an AnnData object.

    Args:
        adata: AnnData object
        genes: list of gene names
        gene_name_key: key in adata.var with gene names

    Returns:
        list of integer indices of genes in adata.var
    """
    series = adata.var.reset_index()[gene_name_key]
    out = []
    for g in genes:
        if g is None:
            continue
        if (type(g) == float) and np.isnan(g):
            continue
        if g not in series.values:
            print(f'{g} not found in the dataset')
            continue
        out.append(series[series == g].idxmax())
    return out


def create_perturbed_dataset_for_gene_set(
    adata, 
    gene_set: list[str], 
    fraction_of_set_to_perturb: float = 0.5, 
    n_perturbations: int = 1000, 
    perturbation_scale: float = 0.1,
    square_epsilon: bool = False,
    cell_perturbations_coherent: bool = True,
    seed: int = 0,
) -> tuple[anndata.AnnData, np.ndarray, np.ndarray]:
    
    assert len(adata) == 1, 'only give one cell to create_perturbed_dataset_for_gene_set'

    # translate to human and see how many remain
    gene_set = [g.upper() for g in gene_set]
    for g in gene_set:
        if g not in adata.var['gene_name'].values:
            print(f'{g} not found in the dataset')
    gene_set = [g for g in gene_set if g in adata.var['gene_name'].values]

    np.random.seed(seed)
    gene_set_subset = np.random.choice(
        gene_set, 
        size=int(len(gene_set) * fraction_of_set_to_perturb), 
        replace=False,
    )
    genes_to_perturb = get_gene_integer_indices(adata, gene_set_subset)
    print('\ngenes perturbed:')
    print('\n'.join(gene_set_subset))
    print('\nother genes in set:')
    print('\n'.join([g for g in gene_set if g not in gene_set_subset]))

    return create_perturbed_dataset(
        adata, 
        n_perturbations=n_perturbations, 
        scale_per_gene_g=np.array(adata.X.mean(axis=0)).squeeze() + 1., 
        genes_to_perturb=genes_to_perturb,
        perturbation_scale=perturbation_scale,
        square_epsilon=square_epsilon,
        cell_perturbations_coherent=cell_perturbations_coherent,
        seed=seed,
    ), gene_set_subset, np.array([g for g in gene_set if g not in gene_set_subset])


def scale_manually(
        X: sp.csr_matrix, 
        mean: np.ndarray, 
        std: np.ndarray, 
        zero_center: bool = False,
        max_value: float | None = None,
    ) -> sp.csr_matrix:
    """
    Scale the sparse matrix X by an input mean and std (rather than computing those from data).

    Args:
        X: sparse matrix to scale
        mean: mean to scale by
        std: std to scale by
        zero_center: whether to zero center the data
        max_value: maximum value to clip the data to

    Returns:
        scaled sparse matrix
    """
    std[std == 0] = 1
    if zero_center:
        X -= mean

    X = axis_mul_or_truediv(
        X,
        std,
        op=truediv,
        out=X if isinstance(X, np.ndarray) or sp.issparse(X) else None,
        axis=1,
    )

    # do the clipping
    if max_value is not None:
        if zero_center:
            a_min, a_max = -max_value, max_value
            X = np.clip(X, a_min, a_max)  # dask does not accept these as kwargs
        else:
            if sp.issparse(X):
                X.data[X.data > max_value] = max_value
            else:
                X[X > max_value] = max_value
    return X


def analyze(
    adata_out, 
    mean_scale_g: np.ndarray, 
    std_scale_g: np.ndarray, 
    log: bool = True, 
    layer: str = 'gpt', 
    max_scale: float = 10.,
):
    """
    Defunct implementation of PCA based on manual scaling of each gene.
    """
    adata_out.X = adata_out.layers[layer].copy()
    if log:
        sc.pp.log1p(adata_out)
    adata_out.X = scale_manually(adata_out.X, mean=mean_scale_g, std=std_scale_g)
    adata_out.X.data = np.clip(np.nan_to_num(adata_out.X.data), -1 * max_scale, max_scale)
    sc.tl.pca(adata_out, n_comps=50)
    return adata_out


def compute_ica_variance_explained(ica: FastICA, data: np.ndarray) -> np.ndarray:
    """
    Compute variances explained by each independent component.
    This is a bit tricky and not 100% sure it's correct.

    Args:
        ica: FastICA object
        data: data to compute explained variance for

    Returns:
        array of explained variance
    """

    explained_variance = []
    for i in range(ica.components_.shape[0]):
        ic_rep = ica.transform(data)
        ic_rep[:, (i + 1):] = 0
        reconstructed_data = ica.inverse_transform(ic_rep)
        explained_variance.append(explained_variance_score(y_true=data, y_pred=reconstructed_data))
    explained_variance = np.array(explained_variance)
    frac_explained_variance = explained_variance / explained_variance[-1]
    frac_explained_variance = np.hstack([frac_explained_variance[0], np.diff(frac_explained_variance)])
    return frac_explained_variance


def analyze_lfc(
    adata_out, 
    pert_layer_key: str = 'perturbed_gpt', 
    measured_layer_key: str = 'measured_gpt', 
    n_pcs: int = 20,
    n_ics: int = 5,
):
    """
    Perform PCA and ICA on log fold change data.

    Args:
        adata_out: AnnData object with log fold change data
        pert_layer_key: key in adata_out.layers with perturbed data
        measured_layer_key: key in adata_out.layers with measured data
        n_pcs: number of principal components
        n_ics: number of independent components

    Returns:
        AnnData object with PCA and ICA components stored in adata_out.varm
    """
    adata_out.X = np.asarray(np.log2( adata_out.layers[pert_layer_key] / adata_out.layers[measured_layer_key] ))

    sc.tl.pca(adata_out, n_comps=n_pcs)

    ica = FastICA(n_components=n_ics, random_state=0, whiten='unit-variance')
    ica.fit(adata_out.X)
    adata_out.varm['ICs'] = ica.components_.T
    ica_explained_variances = compute_ica_variance_explained(ica, adata_out.X)
    adata_out.uns['ica'] = {'variance_ratio': ica_explained_variances}

    return adata_out


def get_loadings(adata, varm_key, component: int = 0):
    """
    Get a dataframe of gene loadings for a component, sorted by square of loading.

    Args:
        adata: AnnData object
        varm_key: loadings are in adata.varm[varm_key]
        component: component to get loadings for

    Returns:
        DataFrame with gene names, loadings, and loading signs
    """
    loadings = adata.varm[varm_key].T
    gene_loading = loadings[component, :].copy()
    gene_power = np.square(gene_loading)
    df = pd.DataFrame({
        'gene_name': adata.var["gene_name"].values,
        'gene_loading': gene_loading,
        'gene_power': gene_power,
        'loading_sign': np.sign(loadings[component, :]),
    })
    df = df.sort_values(by="gene_power", ascending=False)
    df = df[~df['gene_name'].isnull()].copy()
    return df


def get_ic_loadings(adata_out, ic_number: int = 0):
    return get_loadings(adata_out, varm_key='ICs', component=ic_number)


def get_pc_loadings(adata_out, pc_number: int = 0):
    return get_loadings(adata_out, varm_key='PCs', component=pc_number)


class NullIO(io.StringIO):
    def write(self, txt):
        pass


def silent(fn):
    """Decorator to silence functions."""
    def silent_fn(*args, **kwargs):
        with redirect_stdout(NullIO()):
            return fn(*args, **kwargs)
    return silent_fn


def snap_noised_data_to_manifold_and_analyze(
    adata_perturbed: anndata.AnnData,
    pipeline: CellariumPipeline,
    highly_expressed_gene_inds: torch.LongTensor,
    measured_gpt_uns_key: str = 'measured_gpt',
    **analyze_kwargs,
):
    """
    Take a noisy dataset, snap it to the manifold by running the model, and analyze 
    the log fold changes using PCA and ICA.

    Args:
        adata_perturbed_set: AnnData object with perturbed data
        pipeline: CellariumPipeline object
        highly_expressed_gene_inds: indices of highly expressed genes
        **analyze_kwargs: keyword arguments for analyze_lfc, such as n_pcs and n_ics

    Returns:
        AnnData object with PCA and ICA components stored in adata_out.varm
    """

    assert measured_gpt_uns_key in adata_perturbed.uns.keys(), \
        f'measured_gpt_uns_key "{measured_gpt_uns_key}" must be precomputed in adata_perturbed.uns.keys(), try:\n' \
        """\tadata.uns['measured_gpt'] = snap_measured_data_to_manifold(
            adata=adata, 
            pipeline=pipeline, 
            highly_expressed_gene_inds=highly_expressed_gene_inds,
        )"""

    # put the perturbed dataset through the pipeline
    adata_perturbed_set_out = silent(gpt_predict)(
        adata_perturbed, 
        pipeline=pipeline, 
        gene_inds=highly_expressed_gene_inds,
        layer='perturbed',
        key_added='perturbed_gpt',
    )
    adata_perturbed_set_out.layers['measured_gpt'] = sp.vstack([adata_perturbed.uns['measured_gpt']] * len(adata_perturbed_set_out))

    # do PCA and ICA
    adata_perturbed_set_out = analyze_lfc(
        adata_perturbed_set_out, 
        measured_layer_key='measured_gpt',
        **analyze_kwargs,
    )
    return adata_perturbed_set_out


def snap_measured_data_to_manifold(
    adata: anndata.AnnData,
    pipeline: CellariumPipeline,
    highly_expressed_gene_inds: torch.LongTensor,
) -> sp.csr_matrix:
    """
    Snap the measured data to the GPT manifold.

    Args:
        adata: AnnData object with a single cell
        pipeline: CellariumPipeline object
        highly_expressed_gene_inds: indices of highly expressed genes

    Returns:
        sparse matrix with predicted on-manifold expression (floats)
    """

    assert len(adata) == 1, 'only give one cell to snap_measured_data_to_manifold'
    assert 'measured' in adata.layers.keys(), '"measured" must be in adata.layers'

    # put the measured dataset through the pipeline
    adata_out = silent(gpt_predict)(
        adata,
        pipeline=pipeline, 
        gene_inds=highly_expressed_gene_inds,
        layer='measured',
        key_added='measured_gpt',
    )
    return adata_out.layers['measured_gpt']


def noise_prompt_gene_set(
    adata: anndata.AnnData, 
    pipeline: CellariumPipeline,
    gene_set: list[str],
    fraction_of_set_to_perturb: float = 0.5, 
    n_perturbations: int = 1000, 
    perturbation_scale: float = 1.0,
    var_key_include_genes: str = 'gpt_include',
    var_gene_names: str = 'gene_name',
    seed: int = 0,
    **analyze_kwargs,
):
    """
    Perform noise prompting on a gene set, optionally targeting a random fraction of the gene set.

    Args:
        adata: AnnData object with a single cell
        pipeline: CellariumPipeline object
        gene_set: list of gene names to perturb
        fraction_of_set_to_perturb: fraction of gene set to perturb
        n_perturbations: number of perturbations to perform
        perturbation_scale: scale of the perturbation
        var_key_include_genes: key in adata.var to use for gene inclusion
        var_gene_names: key in adata.var to use for gene names
        seed: random seed
        **analyze_kwargs: keyword arguments for analyze_lfc, such as n_pcs and n_ics

    Returns:
        dictionary with keys 'adata', 'genes_perturbed', and 'genes_not_perturbed'
        where output adata has PCA and ICA components stored in adata.varm
    """
    
    assert len(adata) == 1, 'only give one cell to noise_prompt_gene_set'
    assert var_key_include_genes in adata.var.keys(), \
        f'var_key_include_genes "{var_key_include_genes}" must be in adata.var.keys()'
    assert var_gene_names in adata.var.keys(), \
        f'var_gene_names "{var_gene_names}" must be in adata.var.keys()'

    # limit gene set to genes going through gpt
    highly_expressed_gene_inds = torch.tensor(np.where(adata.var[var_key_include_genes])[0]).long()
    highly_expressed_gene_names_set = set(adata.var[var_gene_names].values[adata.var[var_key_include_genes]])
    gene_set = [g for g in gene_set if g in highly_expressed_gene_names_set]

    if len(gene_set) * fraction_of_set_to_perturb < 1:
        raise ValueError(f'gene set too small to perturb: {len(gene_set)} genes, {fraction_of_set_to_perturb} fraction')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # perturb a fraction of the genes in the set
        adata_perturbed_set, genes_perturbed, genes_not_perturbed = silent(create_perturbed_dataset_for_gene_set)(
            adata, 
            gene_set=gene_set,
            fraction_of_set_to_perturb=fraction_of_set_to_perturb, 
            n_perturbations=n_perturbations, 
            perturbation_scale=perturbation_scale,
            square_epsilon=False,
            cell_perturbations_coherent=True,
            seed=seed,
        )

        # perform core of noise prompting analysis
        adata_perturbed_set_out = snap_noised_data_to_manifold_and_analyze(
            adata_perturbed_set,
            pipeline=pipeline,
            highly_expressed_gene_inds=highly_expressed_gene_inds,
            **analyze_kwargs,
        )

    return {
        'adata': adata_perturbed_set_out, 
        'genes_perturbed': genes_perturbed, 
        'genes_not_perturbed': genes_not_perturbed,
    }


def noise_prompt_random(
    adata: anndata.AnnData, 
    pipeline: CellariumPipeline,
    n_perturbations: int = 1000, 
    perturbation_scale: float = 1.0,
    var_key_include_genes: str = 'gpt_include',
    var_key_gene_scale: str = 'mean',
    gene_scale_eps: float = 1e-5,
    seed: int = 0,
    **analyze_kwargs,
) -> anndata.AnnData:
    """
    Perform random noise prompting on all genes.

    Args:
        adata: AnnData object with a single cell
        pipeline: CellariumPipeline object
        n_perturbations: number of perturbations to perform
        perturbation_scale: scale of the perturbation
        var_key_include_genes: key in adata.var to use for gene inclusion
        var_key_gene_scale: key in adata.var with gene scales
        gene_scale_eps: epsilon to add to gene scales
        seed: random seed
        **analyze_kwargs: keyword arguments for analyze_lfc, such as n_pcs and n_ics

    Returns:
        AnnData object with PCA and ICA components stored in adata_out.varm
    """
    assert len(adata) == 1, 'only give one cell to noise_prompt_random'
    assert var_key_include_genes in adata.var.keys(), \
        f'var_key_include_genes "{var_key_include_genes}" must be in adata.var.keys()'
    assert var_key_gene_scale in adata.var.keys(), \
        f'var_key_gene_scale "{var_key_gene_scale}" must be in adata.var.keys()'
    assert gene_scale_eps > 0, 'gene_scale_eps must be positive'

    # limit gene set to genes going through gpt
    highly_expressed_gene_inds = torch.tensor(np.where(adata.var[var_key_include_genes])[0]).long()
    adata.layers['measured'] = adata.X.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata.uns['measured_gpt'] = snap_measured_data_to_manifold(
            adata=adata, 
            pipeline=pipeline, 
            highly_expressed_gene_inds=highly_expressed_gene_inds,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # perturb all genes according to their scales
        adata_perturbed = silent(create_perturbed_dataset)(
            adata,
            n_perturbations=n_perturbations,
            scale_per_gene_g=adata.var[var_key_gene_scale].values + gene_scale_eps,
            perturbation_scale=perturbation_scale,
            seed=seed,
        )

        # perform core of noise prompting analysis
        adata_perturbed_out = snap_noised_data_to_manifold_and_analyze(
            adata_perturbed,
            pipeline=pipeline,
            highly_expressed_gene_inds=highly_expressed_gene_inds,
            **analyze_kwargs,
        )

    return adata_perturbed_out


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


# @background
def compute_pc_set_enrichment(
    adata: anndata.AnnData, 
    genes_perturbed: list[str],
    genes_not_perturbed: list[str],
    pc_number: int, 
    gsea_n_perm: int = 1000,
    seed: int = 0,
    rank_statistic: str = 'gene_loading',
    gene_name_key: str = 'gene_name',
) -> pd.DataFrame:

    df = get_pc_loadings(adata, pc_number=pc_number)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # compute a p-value for the perturbed genes in the set
        gsea_perturbed_stats = silent(gsea)(
            df=df, 
            gene_group=genes_perturbed, 
            gene_name_key=gene_name_key, 
            value_key=rank_statistic,
            n_perm=gsea_n_perm,
            seed=seed,
        )

        # compute a p-value for the unperturbed genes in the set
        gsea_not_perturbed_stats = silent(gsea)(
            df=df, 
            gene_group=genes_not_perturbed, 
            gene_name_key=gene_name_key, 
            value_key=rank_statistic,
            n_perm=gsea_n_perm,
            seed=seed,
        )

    # add to list of dataframes
    data = {
        'pval_perturbed': gsea_perturbed_stats['pval'],
        'es_perturbed': gsea_perturbed_stats['es'],
        'pval_unperturbed': gsea_not_perturbed_stats['pval'],
        'es_unperturbed': gsea_not_perturbed_stats['es'],
        'pc': pc_number,
        'genes_perturbed': [genes_perturbed],
        'genes_not_perturbed': [genes_not_perturbed],
    }
    
    return pd.DataFrame(data=data)


@background
def atomic_write(dfs: list[pd.DataFrame], path: str):
    with tempfile.TemporaryDirectory() as d:
        tmpfile = os.path.join(d, "tmp.csv")
        pd.concat(dfs, axis=0).to_csv(tmpfile, index=False)
        os.replace(tmpfile, path)


def noise_prompt_gene_set_collection(
    adata: anndata.AnnData,
    pipeline: CellariumPipeline,
    msigdb: GeneSetRecords,
    collection: str = 'C5:GO:BP',
    fraction_of_set_to_perturb: float = 0.5, 
    n_random_splits: int = 3,
    n_perturbations: int = 1000, 
    perturbation_scale: float = 1.0,
    min_gene_set_length: int = 10,
    max_gene_set_length: int = 200,
    n_pcs_in_output: int = 5,
    var_key_include_genes: str = 'gpt_include',
    var_gene_names: str = 'gene_name',
    gsea_n_perm: int = 1000,
    seed: int = 0,
    add_random_controls: bool = True,
    save_intermediates_to_tmp_file: str | None = None,
    verbose: bool = True,
    **analyze_kwargs,
):
    """
    Run noise prompting on an entire gene set collection.
    For each gene set, perturb a fraction of the genes and compute PCs.
    For some number of PCs, use the loadings to compute set enrichment for 
    the perturbed and unperturbed genes. Store the p-values and enrichment scores.

    Args:
        adata: AnnData object with a single cell
        pipeline: CellariumPipeline object
        msigdb: GeneSetRecords object
        collection: name of the gene set collection
        fraction_of_set_to_perturb: fraction of gene set to perturb
        n_random_splits: number of times to choose perturbed genes from the set
        n_perturbations: number of perturbations to perform
        perturbation_scale: scale of the perturbation
        min_gene_set_length: minimum gene set length
        max_gene_set_length: maximum gene set length
        n_pcs_in_output: number of PCs to compute
        var_key_include_genes: key in adata.var to use for gene inclusion
        var_gene_names: key in adata.var with gene names
        gsea_n_perm: number of permutations for GSEA p-value computation
        seed: random seed
        add_random_controls: True to add random control gene sets
        save_intermediates_to_tmp_file: path to save intermediate results, and resume from checkpoint
        verbose: True to print out each gene set name
        **analyze_kwargs: keyword arguments for analyze_lfc, such as n_pcs and n_ics

    Returns:
        DataFrame with gene set names, p-values, and enrichment scores
    """
    assert len(adata) == 1, 'only give one cell to noise_prompt_gene_set_collection'
    assert var_key_include_genes in adata.var.keys(), \
        f'var_key_include_genes "{var_key_include_genes}" must be in adata.var.keys()'
    assert var_gene_names in adata.var.keys(), \
        f'var_gene_names "{var_gene_names}" must be in adata.var.keys()'

    np.random.seed(seed)
    
    # append some fake sets for testing purposes
    if add_random_controls:
        control_collection_name = 'random_controls'
        if control_collection_name not in msigdb.get_collections():
            append_random_control_collection(
                msigdb=msigdb, 
                gene_names=adata.var[var_gene_names].values[adata.var[var_key_include_genes] & ~adata.var[var_gene_names].isnull()],
                sizes=[10, 50, 100],
                repeats=5,
                collection_name=control_collection_name,
            )

    # keep track of results in a list of dataframes
    dfs = []
    gene_sets_completed = set()
    if save_intermediates_to_tmp_file is not None:
        if os.path.exists(save_intermediates_to_tmp_file) and (os.path.getsize(save_intermediates_to_tmp_file) > 0):
            dfs.append(pd.read_csv(save_intermediates_to_tmp_file))  # pick up from checkpoint
            gene_sets_completed.update(dfs[-1]['gene_set_name'].unique().tolist())

    # figure out up front which sets will be included based on cutoffs
    highly_expressed_gene_names_set = set(adata.var[var_gene_names][adata.var[var_key_include_genes]].values)

    if add_random_controls:
        all_gene_set_names = msigdb.get_gene_set_names(control_collection_name) + msigdb.get_gene_set_names(collection)
    else:
        all_gene_set_names = msigdb.get_gene_set_names(collection)

    all_gene_set_names = [s for s in all_gene_set_names if s not in gene_sets_completed]
    subset_gene_set_names = []
    for gene_set_name in all_gene_set_names:
        gene_set = msigdb.get_gene_set_dict().get(gene_set_name, [])
        gene_set = [g for g in gene_set if g in highly_expressed_gene_names_set]
        if (len(gene_set) >= min_gene_set_length) and (len(gene_set) <= max_gene_set_length):
            subset_gene_set_names.append(gene_set_name)

    if verbose:
        print(f'gene sets to analyze: {len(subset_gene_set_names)}')

    # compute the measured data snapped to manifold one time
    highly_expressed_gene_inds = torch.tensor(np.where(adata.var[var_key_include_genes])[0]).long()
    adata.layers['measured'] = adata.X.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata.uns['measured_gpt'] = snap_measured_data_to_manifold(
            adata=adata, 
            pipeline=pipeline, 
            highly_expressed_gene_inds=highly_expressed_gene_inds,
        )

    # allow keyboard interrupts
    try:

        # for each gene set
        for gene_set_name in tqdm(subset_gene_set_names):

            if verbose:
                print(gene_set_name)

            # get gene set
            gene_set = msigdb.get_gene_set_dict().get(gene_set_name, [])
            gene_set = [g for g in gene_set if g in highly_expressed_gene_names_set]

            for i in range(n_random_splits):

                # perturb a fraction of the genes in the set
                out = noise_prompt_gene_set(
                    adata,
                    pipeline=pipeline,
                    gene_set=gene_set,
                    fraction_of_set_to_perturb=fraction_of_set_to_perturb,
                    n_perturbations=n_perturbations,
                    perturbation_scale=perturbation_scale,
                    var_key_include_genes=var_key_include_genes,
                    var_gene_names=var_gene_names,
                    seed=i,
                    **analyze_kwargs,
                )
                pc_var_ratios = out['adata'].uns['pca']['variance_ratio']

                # get the loadings of genes on different PCs and compute set enrichment
                for pc_number in range(n_pcs_in_output):
                    df_tmp = compute_pc_set_enrichment(
                        adata=out['adata'],
                        genes_perturbed=out['genes_perturbed'],
                        genes_not_perturbed=out['genes_not_perturbed'],
                        pc_number=pc_number,
                        rank_statistic='gene_power',
                        gene_name_key=var_gene_names,
                        gsea_n_perm=gsea_n_perm,
                        seed=i,
                    )
                    df_tmp['pc_frac_variance_explained'] = pc_var_ratios[pc_number]
                    df_tmp['gene_set_name'] = gene_set_name
                    dfs.append(df_tmp)

            # once all splits and PCs are done (i.e. set is complete), save if called for
            if save_intermediates_to_tmp_file is not None:
                # atomic write
                atomic_write(dfs, save_intermediates_to_tmp_file)

    except KeyboardInterrupt:
        pass

    # package results in a big dataframe
    df = pd.concat(dfs, axis=0)
    return df
