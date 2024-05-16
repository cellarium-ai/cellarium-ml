"""Plotting utilities"""

import numpy as np
import matplotlib.pyplot as plt


def plot_loading(adata, varm_key: str, component: int = 0, cutoff: float = 0.002):
    """
    Plot of loadings of each gene on some component stored in adata.varm[varm_key]

    Args:
        adata: AnnData object
        varm_key: key in adata.varm where the component loadings are stored
        component: component to plot
        cutoff: draw cutoff line at this value

    Returns:
        None
    """

    loadings = adata.varm[varm_key].T

    gene_power = np.square(loadings[component, :])

    for sign in [1, -1]:
        logic = (loadings[component, :] * sign) > 0
        plt.plot(np.arange(len(gene_power))[logic], gene_power[logic], '.', ms=5, 
                label=('positive' if sign == 1 else 'negative'))
    plt.axhline(cutoff, color='r', linestyle='--')
    plt.xlabel('gene')
    plt.ylabel('square of gene loading in component')
    plt.title(f'{varm_key} component {component} gene loadings')
    plt.legend()
    plt.show()


def plot_ic(adata_out, ic_number: int = 0, cutoff: float = 0.002):
    plot_loading(adata_out, varm_key='ICs', component=ic_number, cutoff=cutoff)


def plot_pc(adata_out, pc_number: int = 0, cutoff: float = 0.002):
    plot_loading(adata_out, varm_key='PCs', component=pc_number, cutoff=cutoff)
