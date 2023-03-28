# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Iterable, Optional, Tuple, Union

import pyro
import pyro.distributions as dist
import pytorch_lightning as pl
import torch
from pyro.nn import PyroModule, PyroParam, pyro_method
from torch.distributions import constraints

_PROBABILISTIC_PCA_PYRO_MODULE_NAME = "probabilistic_pca"


class ProbabilisticPCAPyroModule(PyroModule):
    """
    Probabilistic PCA implemented in Pyro.

    **Reference:**

        1. *Probabilistic Principal Component Analysis*,
           Tipping, Michael E., and Christopher M. Bishop. 1999.
           (https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf)

    Args:
        n_cells: Number of cells.
        g_genes: Number of genes.
        k_components: Number of principal components.
        ppca_flavor: Type of the PPCA model. Has to be one of `marginalized` or `diagonal_normal`
            or `multivariate_normal`.
        mean_g: Mean gene expression of the input data.
        w: Scale of the random initialization of the `W_kg` parameter.
        s: Initialization value of the `sigma` parameter.
    """

    def __init__(
        self,
        n_cells: int,
        g_genes: int,
        k_components: int,
        ppca_flavor: str,
        mean_g: Optional[Union[float, int, torch.Tensor]] = None,
        w: float = 1.0,
        s: float = 1.0,
        seed: int = 0,
        transform: Optional[torch.nn.Module] = None,
    ):
        super().__init__(_PROBABILISTIC_PCA_PYRO_MODULE_NAME)

        self.n_cells = n_cells
        self.g_genes = g_genes
        self.k_components = k_components
        assert ppca_flavor in [
            "marginalized",
            "diagonal_normal",
            "multivariate_normal",
        ], "ppca_flavor must be one of 'marginalized' or 'diagonal_normal' or 'multivariate_normal'"
        self.ppca_flavor = ppca_flavor
        self.transform = transform

        if isinstance(mean_g, torch.Tensor) and mean_g.dim():
            assert mean_g.shape == (
                g_genes,
            ), "Expected meang_g to have a shape ({g_genes},) but found {mean_g.shape}."
        if mean_g is None:
            # make mean_g a learnable parameter
            self.mean_g = PyroParam(lambda: torch.zeros(g_genes))
        else:
            self.mean_g = mean_g

        rng = torch.Generator()
        rng.manual_seed(seed)
        # model parameters
        self.W_kg = PyroParam(
            lambda: w * torch.randn((k_components, g_genes), generator=rng)
        )
        self.sigma = PyroParam(lambda: torch.tensor(s), constraint=constraints.positive)

        # guide parameters
        if ppca_flavor == "diagonal_normal":
            M_inv_kk = torch.linalg.inv(self.M_kk)
            z_scale_k_init = torch.sqrt(torch.diag(self.sigma**2 * M_inv_kk)).detach()
            self.z_scale_k = PyroParam(
                lambda: z_scale_k_init, constraint=constraints.positive
            )

    @property
    def M_kk(self):
        return self.W_kg @ self.W_kg.T + self.sigma**2 * torch.eye(
            self.k_components, device=self.sigma.device
        )

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Iterable, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    @pyro_method
    def model(self, x_ng: torch.Tensor) -> None:
        if self.transform is not None:
            x_ng = self.transform(x_ng)

        with pyro.plate("cells", size=self.n_cells, subsample_size=x_ng.shape[0]):
            if self.ppca_flavor == "marginalized":
                pyro.sample(
                    "counts",
                    dist.LowRankMultivariateNormal(
                        loc=self.mean_g,
                        cov_factor=self.W_kg.T,
                        cov_diag=self.sigma**2 * x_ng.new_ones(self.g_genes),
                    ),
                    obs=x_ng,
                )
            else:
                z_nk = pyro.sample(
                    "z",
                    dist.Normal(x_ng.new_zeros(self.k_components), 1).to_event(1),
                )
                pyro.sample(
                    "counts",
                    dist.Normal(self.mean_g + z_nk @ self.W_kg, self.sigma).to_event(1),
                    obs=x_ng,
                )

    @pyro_method
    def guide(self, x_ng: torch.Tensor) -> None:
        if self.ppca_flavor == "marginalized":
            return

        if self.transform is not None:
            x_ng = self.transform(x_ng)

        with pyro.plate("cells", size=self.n_cells, subsample_size=x_ng.shape[0]):
            WX_kn = self.W_kg @ (x_ng - self.mean_g).T
            z_loc_nk = torch.linalg.solve(self.M_kk, WX_kn).T

            if self.ppca_flavor == "diagonal_normal":
                pyro.sample("z", dist.Normal(z_loc_nk, self.z_scale_k).to_event(1))
            elif self.ppca_flavor == "multivariate_normal":
                M_inv_kk = torch.linalg.inv(self.M_kk)
                # This function skips the (slow) error checking and error message construction
                # of torch.linalg.cholesky(), instead directly returning the LAPACK error codes
                # as part of a named tuple (L, info).
                z_scale_tril_kk, _ = torch.linalg.cholesky_ex(
                    self.sigma**2 * M_inv_kk
                )
                pyro.sample(
                    "z",
                    dist.MultivariateNormal(z_loc_nk, scale_tril=z_scale_tril_kk),
                )
            else:
                raise ValueError(
                    "ppca_flavor must be one of 'marginalized' or 'diagonal_normal' or 'multivariate_normal'"
                )

    @torch.inference_mode()
    def get_latent_representation(
        self,
        x_ng: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the latent representation for each cell.
        """
        WX_kn = self.W_kg @ (x_ng - self.mean_g).T
        z_loc_nk = torch.linalg.solve(self.M_kk, WX_kn).T
        return z_loc_nk

    def log(self, plan: pl.LightningModule) -> None:
        W_kg = self.W_kg.detach()
        sigma = self.sigma.detach()
        var_explained = torch.sum(torch.diag(W_kg.T @ W_kg) + sigma**2)
        plan.log("var_explained", var_explained)
