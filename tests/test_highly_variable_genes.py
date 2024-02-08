# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from cellarium.ml.preprocessing.highly_variable_genes import get_highly_variable_genes


@pytest.mark.parametrize("num_genes_to_check", [2, 3, 4])
def test_highly_variable_genes_top_n(num_genes_to_check: int):
    """
    Test if :func:`get_highly_variable_genes` returns the top n genes.
    """
    # data
    gene_names = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4", "gene_5"]
    mean = torch.tensor([1, 2, 3, 4, 5, 6])
    var = torch.tensor(
        [
            1.6086e00,
            2.2582e02,
            9.8922e-02,
            1.4379e01,
            4.0901e-02,
            9.9200e-01,
        ]
    )
    # compute
    result = get_highly_variable_genes(
        gene_names=gene_names, mean=mean, var=var, n_top_genes=num_genes_to_check, n_bins=1
    )
    # assert
    assert result[result.highly_variable].shape[0] == num_genes_to_check


def test_highly_variable_genes_cutoffs():
    """
    Test if :func:`get_highly_variable_genes` returns the genes within the cutoffs.
    """
    # data
    gene_names = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4", "gene_5"]
    mean = torch.tensor([1, 2, 3, 4, 5, 6])
    var = torch.tensor(
        [
            1.6086e00,
            2.2582e02,
            9.8922e-02,
            1.4379e01,
            4.0901e-02,
            9.9200e-01,
        ]
    )
    # compute
    result = get_highly_variable_genes(
        gene_names, mean, var, min_disp=0.2, max_disp=10, min_mean=0, max_mean=5, n_bins=1
    )
    # assert
    assert result[result.highly_variable].shape[0] != 0


def test_highly_variable_genes_wrong_sizes():
    """
    Test if :func:`get_highly_variable_genes` raises an error when the sizes of the inputs are different.
    """
    # data
    gene_names = ["gene_0", "gene_1", "gene_2", "gene_3"]
    mean = torch.tensor([1, 2, 3])
    var = torch.tensor([0.1, 0.2, 0.3, 0.4])
    # assert
    with pytest.raises(ValueError):
        get_highly_variable_genes(gene_names, mean, var)
