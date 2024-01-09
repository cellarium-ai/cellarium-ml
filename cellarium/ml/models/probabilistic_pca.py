# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroParam
from torch.distributions import constraints

from cellarium.ml.models.model import CellariumModel, PredictMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class ProbabilisticPCA(CellariumModel, PredictMixin):
    """
    Probabilistic PCA implemented in Pyro.

    Two flavors of probabilistic PCA are available - marginalized pPCA [1]
    and linear VAE [2].

    **References:**

    1. `Probabilistic Principal Component Analysis (Tipping et al.)
       <https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf>`_.
    2. `Understanding Posterior Collapse in Generative Latent Variable Models (Lucas et al.)
       <https://openreview.net/pdf?id=r1xaVLUYuE>`_.

    Args:
        n_obs:
            Number of cells.
        feature_schema:
            The variable names schema for the input data validation.
        k_components:
            Number of principal components.
        ppca_flavor:
            Type of the PPCA model. Has to be one of `marginalized` or `linear_vae`.
        mean_g:
            Mean gene expression of the input data. If ``None`` then the mean is
            set to a learnable parameter.
        W_init_scale:
            Scale of the random initialization of the `W_kg` parameter.
        sigma_init_scale:
            Initialization value of the `sigma` parameter.
        seed:
            Random seed used to initialize parameters.
        elbo:
            ELBO loss function. Should be a subclass of :class:`~pyro.infer.ELBO`.
            If ``None``, defaults to :class:`~pyro.infer.Trace_ELBO`.
    """

    def __init__(
        self,
        n_obs: int,
        feature_schema: Sequence[str],
        k_components: int,
        ppca_flavor: str,
        mean_g: float | torch.Tensor | None = None,
        W_init_scale: float = 1.0,
        sigma_init_scale: float = 1.0,
        seed: int = 0,
        elbo: pyro.infer.ELBO | None = None,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.feature_schema = np.array(feature_schema)
        g_genes = len(self.feature_schema)
        self.g_genes = g_genes
        self.k_components = k_components
        assert ppca_flavor in [
            "marginalized",
            "linear_vae",
        ], "ppca_flavor must be one of 'marginalized' or 'linear_vae'"
        self.ppca_flavor = ppca_flavor
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

        rng = torch.Generator()
        rng.manual_seed(seed)
        # model parameters
        self.W_kg = PyroParam(lambda: W_init_scale * torch.randn((k_components, g_genes), generator=rng))
        self.sigma = PyroParam(lambda: torch.tensor(sigma_init_scale), constraint=constraints.positive)

    def forward(self, x_ng: torch.Tensor, feature_g: np.ndarray) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng:
                Gene counts matrix.
            feature_g:
                The list of the variable names in the input data.

        Returns:
            A dictionary with the loss value.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng)
        return {"loss": loss}

    def model(self, x_ng: torch.Tensor) -> None:
        with pyro.plate("cells", size=self.n_obs, subsample_size=x_ng.shape[0]):
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

        with pyro.plate("cells", size=self.n_obs, subsample_size=x_ng.shape[0]):
            V_gk = torch.linalg.solve(self.M_kk, self.W_kg).T
            D_k = self.sigma / torch.sqrt(torch.diag(self.M_kk))
            pyro.sample("z", dist.Normal((x_ng - self.mean_g) @ V_gk, D_k).to_event(1))

    def predict(self, x_ng: torch.Tensor, feature_g: np.ndarray) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Centering and embedding of the input data ``x_ng`` into the principal component space.

        .. note::
           Gradients are disabled, used for inference only.

        Args:
            x_ng:
                Gene counts matrix.
            feature_g:
                The list of the variable names in the input data.

        Returns:
            A dictionary with the following keys:

            - ``z_nk``: Embedding of the input data into the principal component space.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "feature_g", feature_g)
        assert_arrays_equal("feature_g", feature_g, "feature_schema", self.feature_schema)

        V_gk = torch.linalg.solve(self.M_kk, self.W_kg).T
        z_nk = (x_ng - self.mean_g) @ V_gk
        return {"z_nk": z_nk}

    @property
    def M_kk(self) -> torch.Tensor:
        return self.W_kg @ self.W_kg.T + self.sigma**2 * torch.eye(self.k_components, device=self.sigma.device)

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
