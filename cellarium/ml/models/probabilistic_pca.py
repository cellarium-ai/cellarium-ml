# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence
from typing import Literal

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.nn.module import PyroParam, _unconstrain
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
        var_names_g:
            The variable names schema for the input data validation.
        n_components:
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
        var_names_g: Sequence[str],
        n_components: int,
        ppca_flavor: Literal["marginalized", "linear_vae"],
        mean_g: torch.Tensor | None = None,
        W_init_scale: float = 1.0,
        sigma_init_scale: float = 1.0,
        seed: int = 0,
        elbo: pyro.infer.ELBO | None = None,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.var_names_g = np.array(var_names_g)
        n_vars = len(self.var_names_g)
        self.n_vars = n_vars
        self.n_components = n_components
        self.ppca_flavor = ppca_flavor
        self.elbo = elbo or pyro.infer.Trace_ELBO()

        if isinstance(mean_g, torch.Tensor) and mean_g.dim():
            assert mean_g.shape == (n_vars,), f"Expected meang_g to have a shape ({n_vars},) but found {mean_g.shape}."
        if mean_g is None:
            # make mean_g a learnable parameter
            self.mean_g = torch.nn.Parameter(torch.empty(n_vars))
        else:
            self.register_buffer("mean_g", mean_g)

        self.seed = seed
        # model parameters
        self.W_init_scale = W_init_scale
        self.sigma_init_scale = sigma_init_scale
        self.W_kg = torch.nn.Parameter(torch.empty(n_components, n_vars))
        self.sigma = PyroParam(torch.empty(()), constraint=constraints.positive)  # type: ignore[call-arg]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        rng_device = self.W_kg.device.type if self.W_kg.device.type != "meta" else "cpu"
        rng = torch.Generator(device=rng_device)
        rng.manual_seed(self.seed)
        if isinstance(self.mean_g, torch.nn.Parameter):
            self.mean_g.data.zero_()
        self.W_kg.data.normal_(0, self.W_init_scale, generator=rng)
        self.sigma_unconstrained.data.fill_(_unconstrain(torch.as_tensor(self.sigma_init_scale), constraints.positive))

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.

        Returns:
            A dictionary with the loss value.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng)  # type: ignore[attr-defined]
        return {"loss": loss}

    def model(self, x_ng: torch.Tensor) -> None:
        with pyro.plate("cells", size=self.n_obs, subsample_size=x_ng.shape[0]):
            if self.ppca_flavor == "marginalized":
                pyro.sample(
                    "counts",
                    dist.LowRankMultivariateNormal(  # type: ignore[attr-defined]
                        loc=self.mean_g,
                        cov_factor=self.W_kg.T,  # type: ignore[attr-defined]
                        cov_diag=self.sigma**2 * x_ng.new_ones(self.n_vars),  # type: ignore[operator]
                    ),
                    obs=x_ng,
                )
            else:
                z_nk = pyro.sample(
                    "z",
                    dist.Normal(x_ng.new_zeros(self.n_components), 1).to_event(1),  # type: ignore[attr-defined]
                )
                pyro.sample(
                    "counts",
                    dist.Normal(self.mean_g + z_nk @ self.W_kg, self.sigma).to_event(1),  # type: ignore[attr-defined]
                    obs=x_ng,
                )

    def guide(self, x_ng: torch.Tensor) -> None:
        if self.ppca_flavor == "marginalized":
            return

        with pyro.plate("cells", size=self.n_obs, subsample_size=x_ng.shape[0]):
            V_gk = torch.linalg.solve(self.M_kk, self.W_kg).T
            D_k = self.sigma / torch.sqrt(torch.diag(self.M_kk))
            pyro.sample("z", dist.Normal((x_ng - self.mean_g) @ V_gk, D_k).to_event(1))  # type: ignore[attr-defined]

    def predict(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Centering and embedding of the input data ``x_ng`` into the principal component space.

        .. note::
           Gradients are disabled, used for inference only.

        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.

        Returns:
            A dictionary with the following keys:

            - ``z_nk``: Embedding of the input data into the principal component space.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        V_gk = torch.linalg.solve(self.M_kk, self.W_kg).T
        z_nk = (x_ng - self.mean_g) @ V_gk
        return {"z_nk": z_nk}

    @property
    def M_kk(self) -> torch.Tensor:
        return self.W_kg @ self.W_kg.T + self.sigma**2 * torch.eye(self.n_components, device=self.sigma.device)  # type: ignore[attr-defined, operator]

    @property
    @torch.inference_mode()
    def L_k(self) -> torch.Tensor:
        r"""
        Vector with elements given by the PC eigenvalues.

        .. note::
           Gradients are disabled, used for inference only.
        """
        S_k = torch.linalg.svdvals(self.W_kg.T)  # type: ignore[attr-defined]
        return S_k**2 + self.sigma**2  # type: ignore[operator]

    @property
    @torch.inference_mode()
    def U_gk(self) -> torch.Tensor:
        r"""
        Principal components corresponding to eigenvalues ``L_k``.

        .. note::
           Gradients are disabled, used for inference only.
        """
        return torch.linalg.svd(self.W_kg.T, full_matrices=False).U  # type: ignore[attr-defined]

    @property
    @torch.inference_mode()
    def W_variance(self) -> float:
        r"""
        .. note::
           Gradients are disabled, used for inference only.
        """
        return torch.trace(self.W_kg.T @ self.W_kg).item()  # type: ignore[attr-defined]

    @property
    @torch.inference_mode()
    def sigma_variance(self) -> float:
        r"""
        .. note::
           Gradients are disabled, used for inference only.
        """
        return (self.n_vars * self.sigma**2).item()  # type: ignore[operator]
