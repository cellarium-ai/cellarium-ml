import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule, PyroParam
from scvi.module.base import PyroBaseModuleClass
from torch.distributions import constraints

_PROBABILISTIC_PCA_PYRO_MODULE_NAME = "probabilistic_pca"


class ProbabilisticPCAPyroModel(PyroModule):
    """
    A PyroModule that serves as the model for the ProbabilisticPCAPyroModule class.

    Args:
        n_obs: Number of observations.
        n_vars: Number of input features.
        n_components: Number of components to model.
        mean: Mean of the input data.
        marginalize_z: Marginalize out latent variable z.
    """

    def __init__(
        self,
        n_obs: int,
        n_vars: int,
        n_components: int,
        mean: torch.Tensor,
        marginalize_z: bool,
    ):
        super().__init__(_PROBABILISTIC_PCA_PYRO_MODULE_NAME)

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_components = n_components
        self.mean = mean
        self.marginalize_z = marginalize_z

        # model parameters
        self.W = PyroParam(lambda: torch.randn((n_components, n_vars)))
        self.sigma = PyroParam(
            lambda: torch.tensor(1.0), constraint=constraints.positive
        )

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        with pyro.plate("cells", size=self.n_obs, subsample_size=x.shape[0]):
            if self.marginalize_z:
                pyro.sample(
                    "counts",
                    dist.LowRankMultivariateNormal(
                        loc=self.mean,
                        cov_factor=self.W.T,
                        cov_diag=self.sigma**2
                        * torch.ones(self.n_vars, device=x.device),
                    ),
                    obs=x,
                )
            else:
                z = pyro.sample(
                    "z",
                    dist.Normal(
                        torch.zeros(self.n_components, device=x.device), 1
                    ).to_event(1),
                )
                pyro.sample(
                    "counts",
                    dist.Normal(self.mean + z @ self.W, self.sigma).to_event(1),
                    obs=x,
                )


class ProbabilisticPCAPyroGuide(PyroModule):
    """
    A PyroModule that serves as the guide for the ProbabilisticPCAPyroModule class.

    Args:
        n_obs: Number of observations.
        n_vars: Number of input features.
        n_components: Number of components to model.
        mean: Mean of the input data.
        marginalize_z: Marginalize out latent variable z.
    """

    def __init__(
        self,
        n_obs: int,
        n_vars: int,
        n_components: int,
        mean: torch.Tensor,
        marginalize_z: bool,
    ):
        super().__init__(_PROBABILISTIC_PCA_PYRO_MODULE_NAME)

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_components = n_components
        self.mean = mean
        self.marginalize_z = marginalize_z

        # guide parameters
        if not self.marginalize_z:
            self.L = PyroParam(lambda: torch.randn((n_vars, n_components)))
            self.z_scale = PyroParam(
                lambda: torch.ones(n_components), constraint=constraints.positive
            )

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        if not self.marginalize_z:
            with pyro.plate("cells", size=self.n_obs, subsample_size=x.shape[0]):
                z_loc = (x - self.mean) @ self.L
                pyro.sample("z", dist.Normal(z_loc, self.z_scale).to_event(1))


class ProbabilisticPCAPyroModule(PyroBaseModuleClass):
    """
    Probabilistic PCA implemented in Pyro.

    Args:
        n_obs: Number of observations.
        n_vars: Number of input features.
        n_components: Number of components to model.
        mean: Mean of the input data.
        marginalize_z: Marginalize out latent variable z.
    """

    def __init__(
        self,
        n_obs: int,
        n_vars: int,
        n_components: int,
        mean: torch.Tensor = 0,
        marginalize_z: bool = True,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_components = n_components
        self.mean = mean
        self.marginalize_z = marginalize_z

        self._model = ProbabilisticPCAPyroModel(
            self.n_obs, self.n_vars, self.n_components, self.mean, self.marginalize_z
        )
        self._guide = ProbabilisticPCAPyroGuide(
            self.n_obs, self.n_vars, self.n_components, self.mean, self.marginalize_z
        )

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @torch.inference_mode()
    def get_latent_representation(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the latent representation for each cell.
        """
        L = self._guide.L
        z_loc = (x - self.mean) @ L
        return z_loc
