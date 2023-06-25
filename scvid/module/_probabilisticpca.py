# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import math
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroParam
from torch.distributions import constraints
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_mv

from .base_module import BasePyroModule


def _batch_lowrank_logdet(W, D, capacitance_tril):
    r"""
    Uses "matrix determinant lemma"::
        log|W @ W.T + D| = log|C| + log|D|,
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute
    the log determinant.
    """
    return 2 * capacitance_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) + D.log().sum(
        -1
    )


def _batch_lowrank_mahalanobis(W, D, x, capacitance_tril, total_var):
    r"""
    Uses "Woodbury matrix identity"::
        inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D),
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute the squared
    Mahalanobis distance :math:`x.T @ inv(W @ W.T + D) @ x`.
    """
    Wt_Dinv = W.mT / D.unsqueeze(-2)
    Wt_Dinv_x = _batch_mv(Wt_Dinv, x)
    # mahalanobis_term_ = (x.pow(2) / D).sum(-1)
    mahalanobis_term1 = (total_var / D).mean(-1)
    mahalanobis_term2 = _batch_mahalanobis(capacitance_tril, Wt_Dinv_x)
    return mahalanobis_term1 - mahalanobis_term2


class ProbabilisticPCA(BasePyroModule):
    """
    Probabilistic PCA implemented in Pyro.

    Two flavors of probabilistic PCA are available - marginalized pPCA [1]
    and linear VAE [2].

    **Reference:**

    1. *Probabilistic Principal Component Analysis*,
       Tipping, Michael E., and Christopher M. Bishop. 1999.
       (https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf)
    2. *Understanding Posterior Collapse in Generative Latent Variable Models*,
       James Lucas, George Tucker, Roger Grosse, Mohammad Norouzi. 2019.
       (https://openreview.net/pdf?id=r1xaVLUYuE)

    Args:
        n_cells: Number of cells.
        g_genes: Number of genes.
        k_components: Number of principal components.
        ppca_flavor: Type of the PPCA model. Has to be one of `marginalized` or `linear_vae`.
        mean_g: Mean gene expression of the input data.
        W_init_scale: Scale of the random initialization of the `W_kg` parameter.
        sigma_init_scale: Initialization value of the `sigma` parameter.
        seed: Random seed used to initialize parameters. Default: ``0``.
        transform: If not ``None`` is used to transform the input data.
        elbo: ELBO loss function. Should be a subclass of :class:`~pyro.infer.ELBO`.
            If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    """

    def __init__(
        self,
        n_cells: int,
        g_genes: int,
        k_components: int,
        ppca_flavor: str,
        mean_g: float | torch.Tensor | None = None,
        W_init_scale: float = 1.0,
        sigma_init_scale: float = 1.0,
        seed: int = 0,
        transform: torch.nn.Module | None = None,
        elbo: pyro.infer.ELBO | None = None,
        S: torch.Tensor | None = None,
        total_var: torch.Tensor | None = None,
    ):
        super().__init__(type(self).__name__)

        self.n_cells = n_cells
        self.g_genes = g_genes
        self.k_components = k_components
        assert ppca_flavor in [
            "marginalized",
            "linear_vae",
        ], "ppca_flavor must be one of 'marginalized' or 'linear_vae'"
        self.ppca_flavor = ppca_flavor
        self.transform = transform
        self.elbo = elbo or pyro.infer.Trace_ELBO()

        if isinstance(mean_g, torch.Tensor) and mean_g.dim():
            assert mean_g.shape == (
                g_genes,
            ), f"Expected meang_g to have a shape ({g_genes},) but found {mean_g.shape}."
        if mean_g is None:
            # make mean_g a learnable parameter
            self.mean_g = PyroParam(lambda: torch.zeros(g_genes))
        else:
            self.register_buffer("mean_g", torch.as_tensor(mean_g))

        if S is None:
            self.S = None
        else:
            self.register_buffer("S", S)

        if total_var is None:
            self.total_var = None
        else:
            self.register_buffer("total_var", total_var)

        rng = torch.Generator()
        rng.manual_seed(seed)
        # model parameters
        self.W_kg = PyroParam(
            lambda: W_init_scale * torch.randn((k_components, g_genes), generator=rng)
        )
        self.sigma = PyroParam(
            lambda: torch.tensor(sigma_init_scale), constraint=constraints.positive
        )

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: dict[str, torch.Tensor]
    ) -> tuple[tuple, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.S is not None:
            low_rank = dist.LowRankMultivariateNormal(
                loc=self.mean_g,
                cov_factor=self.W_kg.T,
                cov_diag=self.sigma**2 * self.W_kg.new_ones(self.g_genes),
            )
            log_det = _batch_lowrank_logdet(
                low_rank._unbroadcasted_cov_factor,
                low_rank._unbroadcasted_cov_diag,
                low_rank._capacitance_tril,
            )
            diff = args[0] - self.mean_g
            M = _batch_lowrank_mahalanobis(
                low_rank._unbroadcasted_cov_factor,
                low_rank._unbroadcasted_cov_diag,
                diff,
                low_rank._capacitance_tril,
                self.total_var,
            )
            cost = (
                -0.5
                # * self.n_cells
                * (
                    self.g_genes * math.log(2 * math.pi)
                    + log_det
                    + M
                    # + torch.sum(self.C_inv * self.S.T)
                )
            ).sum() * self.n_cells / args[0].shape[0]
            return -cost
        return self.elbo.differentiable_loss(self.model, self.guide, *args, **kwargs)

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

    def guide(self, x_ng: torch.Tensor) -> None:
        if self.ppca_flavor == "marginalized":
            return

        if self.transform is not None:
            x_ng = self.transform(x_ng)

        with pyro.plate("cells", size=self.n_cells, subsample_size=x_ng.shape[0]):
            V_gk = torch.linalg.solve(self.M_kk, self.W_kg).T
            D_k = self.sigma / torch.sqrt(torch.diag(self.M_kk))
            pyro.sample("z", dist.Normal((x_ng - self.mean_g) @ V_gk, D_k).to_event(1))

    @torch.inference_mode()
    def get_latent_representation(
        self,
        x_ng: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Return the latent representation for each cell.

        .. note::
           Gradients are disabled, used for inference only.
        """
        V_gk = torch.linalg.solve(self.M_kk, self.W_kg).T
        return (x_ng - self.mean_g) @ V_gk

    @property
    def C_inv(self) -> torch.Tensor:
        #  return (
        #      torch.eye(self.g_genes, device=self.sigma.device) / self.sigma**2
        #      - 1
        #      / self.sigma**2
        #      * self.W_kg.T
        #      @ torch.linalg.inv(self.M_kk)
        #      @ self.W_kg
        #  )
        return (
            -1 / self.sigma**2 * self.W_kg.T @ torch.linalg.inv(self.M_kk) @ self.W_kg
        )

    @property
    def M_kk(self) -> torch.Tensor:
        return self.W_kg @ self.W_kg.T + self.sigma**2 * torch.eye(
            self.k_components, device=self.sigma.device
        )

    @property
    @torch.inference_mode()
    def L_k(self) -> torch.Tensor:
        r"""
        Vector with elements given by the PC eigenvalues.

        .. note::
           Gradients are disabled, used for inference only.
        """
        S_k = torch.linalg.svdvals(self.W_kg.T)
        return S_k**2 + self.sigma**2

    @property
    @torch.inference_mode()
    def U_gk(self) -> torch.Tensor:
        r"""
        Principal components corresponding to eigenvalues ``L_k``.

        .. note::
           Gradients are disabled, used for inference only.
        """
        return torch.linalg.svd(self.W_kg.T, full_matrices=False).U

    @property
    @torch.inference_mode()
    def W_variance(self) -> float:
        r"""
        .. note::
           Gradients are disabled, used for inference only.
        """
        return torch.trace(self.W_kg.T @ self.W_kg).item()

    @property
    @torch.inference_mode()
    def sigma_variance(self) -> float:
        r"""
        .. note::
           Gradients are disabled, used for inference only.
        """
        return (self.g_genes * self.sigma**2).item()
