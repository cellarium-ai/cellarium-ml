# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Iterable, Optional, Tuple, Union

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule, PyroParam, pyro_method
from torch.distributions import constraints

_PROBABILISTIC_PCA_PYRO_MODULE_NAME = "probabilistic_pca"


class ProbabilisticPCAPyroModule(PyroModule):
    """
    Probabilistic PCA implemented in Pyro.

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
        if ppca_flavor == "marginalized":
            pass
        elif ppca_flavor == "diagonal_normal":
            self.L_gk = PyroParam(
                lambda: torch.randn((g_genes, k_components), generator=rng)
            )
            self.z_scale_k = PyroParam(
                lambda: torch.ones(k_components), constraint=constraints.positive
            )
        elif ppca_flavor == "multivariate_normal":
            self.L_gk = PyroParam(
                lambda: torch.randn((g_genes, k_components), generator=rng)
            )
            self.z_scale_tril_kk = PyroParam(
                lambda: torch.eye(k_components),
                constraint=constraints.lower_cholesky,
            )
        else:
            raise ValueError(
                "ppca_flavor must be one of 'marginalized' or 'diagonal_normal' or 'multivariate_normal'"
            )

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Iterable, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    @pyro_method
    def model(self, x_ng: torch.Tensor):
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
    def guide(self, x_ng: torch.Tensor):
        if self.ppca_flavor == "marginalized":
            return

        if self.transform is not None:
            x_ng = self.transform(x_ng)

        with pyro.plate("cells", size=self.n_cells, subsample_size=x_ng.shape[0]):
            z_loc_nk = (x_ng - self.mean_g) @ self.L_gk

            if self.ppca_flavor == "diagonal_normal":
                pyro.sample("z", dist.Normal(z_loc_nk, self.z_scale_k).to_event(1))
            elif self.ppca_flavor == "multivariate_normal":
                pyro.sample(
                    "z",
                    dist.MultivariateNormal(z_loc_nk, scale_tril=self.z_scale_tril_kk),
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
        if self.ppca_flavor == "marginalized":
            M_kk = self.W_kg @ self.W_kg.T + self.sigma**2 * torch.eye(
                self.k_components
            )
            L_gk = self.W_kg.T @ torch.linalg.inv(M_kk)
        elif self.ppca_flavor in ("diagonal_normal", "multivariate_normal"):
            L_gk = self.L_gk
        else:
            raise ValueError(
                "ppca_flavor must be one of 'marginalized' or 'diagonal_normal' or 'multivariate_normal'"
            )
        z_loc_nk = (x_ng - self.mean_g) @ L_gk
        return z_loc_nk
