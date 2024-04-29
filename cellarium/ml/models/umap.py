# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import numpy as np
import torch
import scipy.sparse as sp
from umap import UMAP

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


def distance_to_log_prob(distance: torch.Tensor, a: float = 1.0, b: float = 1.0):
    """
    From `~umap.parametric_umap.convert_distance_to_log_probability`, computes 
    an "edge probability" given a distance between points in the embedding.

    Args:
        distance:
            tensor containing euclidean distances between points in embedding
        a:
            umap parameter based on min_dist
        b:
            umap parameter based on min_dist

    Returns:
        tensor containing log probabilities of edges, same shape as distance
    """
    return -torch.log1p(a * torch.pow(distance, 2 * b))


def umap_loss(
    embedding_nk: torch.Tensor,
    knn_graph: sp.csr_matrix,
    negative_sample_rate: float,
    a: float,
    b: float,
    n_steps_max: int = 200,
):
    """
    Compute the differentiable UMAP loss function.
    
    Inspiration from `~umap.umap_.simplicial_set_embedding` and `~umap.parametric_umap.umap_loss`.

    This is a binary cross-entropy loss that uses truth labels of 1 for edges in the kNN graph and 0 for 
    negative samples (of which there are negative_sample_rate times as many). Probabilities of edges in 
    the embedding are computed based on euclidean distances between embedded points.
    
    """
    assert n_steps_max > 10, "n_steps_max must be greater than 10"
    knn_graph = knn_graph.tocoo()
    knn_graph.sum_duplicates()
    n_vertices = knn_graph.shape[1]

    knn_graph.data[knn_graph.data < (knn_graph.data.max() / float(n_steps_max))] = 0.0
    knn_graph.eliminate_zeros()

    # all the real nearest-neighbor edges in the kNN graph and their weights
    head_n = knn_graph.row
    tail_n = knn_graph.col
    weight_n = knn_graph.data  # TODO: what is the right way to use these weights?

    # randomly sample negative_sample_rate times as many (very probably) non-edges
    neg_head_p = np.random.randint(0, n_vertices, size=int(negative_sample_rate * len(head_n)))
    neg_tail_p = np.random.randint(0, n_vertices, size=int(negative_sample_rate * len(head_n)))

    # put together a long list of indices that define edges of interest
    head_inds_m = torch.from_numpy(np.hstack([head_n, neg_head_p])).long()
    tail_inds_m = torch.from_numpy(np.hstack([tail_n, neg_tail_p])).long()
    weight_m = torch.from_numpy(np.hstack([weight_n, np.ones(len(neg_head_p))])).float()

    # compute their distances in the embedding
    distances_m = torch.norm(embedding_nk[head_inds_m, :] - embedding_nk[tail_inds_m, :], dim=1)

    # convert distances to log probabilities
    log_prob_m = distance_to_log_prob(distance=distances_m, a=a, b=b)

    # tensor describing whether each edge is a true edge (1) or a negative sample (0)
    true_edges_m = torch.cat([torch.ones(len(head_n)), torch.zeros(len(neg_head_p))])

    # compute the binary cross-entropy loss
    # loss = torch.nn.BCELoss(weight=None)(log_prob_m.exp(), true_edges_m)  # weight? stability?
    loss = torch.nn.BCEWithLogitsLoss(weight=weight_m)(log_prob_m, true_edges_m)  # weight?
    # I don't know why, but it sure looks like they're doing "WithLogits" in umap.parametric_umap.compute_cross_entropy

    return loss


class SubsampledApproximateUMAP(UMAP, torch.nn.Module, PredictMixin):
    """
    Scalable model for a UMAP embedding parameterized by an encoder neural network.

    **References:**

    1. `UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. 
       (McInnes, Healy, Melville. 2018) <https://arxiv.org/abs/1802.03426>`_.

    Args:
        var_names_g:
            The variable names schema for the input data validation.
        encoder:
            The encoder neural network `~torch.nn.Module` to use for the UMAP embedding.
        **umap_kwargs:
            Keyword arguments passed to the `~umap.UMAP` constructor.
        
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        encoder: torch.nn.Module,
        **umap_kwargs,
    ) -> None:
        torch.nn.Module.__init__(self)
        UMAP.__init__(self, **umap_kwargs)
        self.var_names_g = np.array(var_names_g)
        self.encoder = encoder
        self._differentiable_loss = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.init_weights()

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, torch.Tensor]:
        """
        For the given minibatch, run UMAP minus the built-in `~umap.UMAP._fit_embed_data` step. 
        Replace this (which iterates to convergence in `~umap.UMAP`) with a single calculation of 
        the loss. (This is later used to update the encoder.)

        Here, the loss is computed based only on the kNN graph of the data passed in this minibatch, 
        thus the approximate nature of this procedure.

        Args:
            x_ng:
                Gene counts matrix or other data embedding (such as PC loadings).
            var_names_g:
                The list of the variable names in the input data.
        Returns:
            A dictionary with the loss value.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        # the call to `~umap.UMAP.fit` calls self._fit_embed_data which stores a loss
        # super(UMAP, self).fit(x_ng)
        self.fit(x_ng)
        return {"loss": self._differentiable_loss}
    
    def _fit_embed_data(self, X, n_epochs, init, random_state) -> tuple[np.ndarray, dict[str, any]]:
        """
        Overrides method in `~umap.UMAP` superclass. Computes the loss rather than iterating 
        on the embedding to convergence.

        self.graph_ is the kNN graph of the data passed in this minibatch, computed by 
        the superclass `~umap.UMAP.fit` method which calls this method.
        """
        x_ng = torch.from_numpy(X)
        embedding_nk = self.encoder(x_ng)

        self._differentiable_loss = umap_loss(
            embedding_nk=embedding_nk,
            knn_graph=self.graph_,
            negative_sample_rate=self.negative_sample_rate,
            a=self._a,
            b=self._b,
            # n_steps_max=self.n_epochs,  # probably wrong
        )
        return embedding_nk.detach().numpy(), {}

    def predict(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> torch.Tensor:
        """
        Encode data with the encoder to create the UMAP embedding.

        Args:
            x_ng:
                Gene counts matrix or other data embedding (such as PC loadings).
            var_names_g:
                The list of the variable names in the input data.

        Returns:
            The UMAP embedding
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        embedding_nk = self.encoder(x_ng)
        return embedding_nk
