# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models import ValidateMixin
from cellarium.ml.models.nmf import (
    NMFInit,
    NMFInitSklearnRandom,
    NMFInitUniformRandom,
    NonNegativeMatrixFactorization,
    compute_reconstruction_error_compiled,
    nmf_frobenius_loss,
    online_dictionary_update_fista,
    online_dictionary_update_nmf_torch_hals,
    solve_nnls_fista,
)
from cellarium.ml.utilities.convergence import NoisyConvergenceTracker
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class FiLMBlock(torch.nn.Module):
    """
    A block containing a shared linear layer, followed by replicate-specific FiLM modulation.
    """

    def __init__(self, input_dim: int, output_dim: int, num_replicates: int):
        super().__init__()

        # shared linear layer
        self.linear = torch.nn.Linear(input_dim, output_dim)

        # batch norm without affine
        self.batch_norm = torch.nn.BatchNorm1d(output_dim, affine=False)

        # replicate-specific FiLM parameters: gamma and beta for each replicate
        self.gamma_rh = torch.nn.Parameter(torch.ones(num_replicates, output_dim))
        self.beta_rh = torch.nn.Parameter(torch.zeros(num_replicates, output_dim))

        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linear layer
        h = self.linear(x)

        # batch norm
        if h.dim() == 3:
            # collapse first two dimensions r and n to apply batch norm, then un-collapse
            r, n, h_dim = h.shape
            h = h.view(r * n, h_dim)
            h_rnh = self.batch_norm(h.view(r * n, h_dim)).view(r, n, h_dim)
        else:
            h_rnh = self.batch_norm(h).unsqueeze(0)  # expand to (1, N, H)

        # expand gamma and beta to broadcast across the batch dimension (n)
        # from (r, output_dim) -> (r, 1, output_dim)
        gamma_r1h = self.gamma_rh.unsqueeze(1)
        beta_r1h = self.beta_rh.unsqueeze(1)

        # apply FiLM modulation
        h_modulated_rnh = gamma_r1h * h_rnh + beta_r1h

        return self.relu(h_modulated_rnh)


class ConsensusNMFEncoder(torch.nn.Module):
    """
    Encoder network to predict NMF loadings from gene expression data, with FiLM modulation
    to handle multiple replicates. This effectively works as if it were num_replicates
    separate encoders.
    """

    def __init__(self, num_genes: int, hidden_dims: list[int], num_factors: int, num_replicates: int):
        super().__init__()
        self.num_replicates = num_replicates
        self.num_factors = num_factors

        self.blocks = torch.nn.ModuleList()
        prev_dim = num_genes
        for hidden_dim in hidden_dims:
            self.blocks.append(FiLMBlock(prev_dim, hidden_dim, num_replicates))
            prev_dim = hidden_dim

        # Final layer to output the K factors.
        # Standard Linear layer applied to the last dimension.
        self.output_layer = torch.nn.Linear(prev_dim, num_factors)

        # NMF loadings must be non-negative, so we cap the network with a Softplus
        self.relu = torch.nn.ReLU()

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: Input gene expression of shape (N, G)
        Returns:
            loadings_rnk: NMF loadings of shape (R, N, K)
        """
        h = x_ng

        for block in self.blocks:
            # block 1 takes (N, G) -> outputs (R, N, H1)
            # block 2+ takes (R, N, H1) -> outputs (R, N, H2)
            h = block(h)

        # final projection to (R, N, K)
        loadings_rnk = self.output_layer(h)

        return self.relu(loadings_rnk)


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, FiLMBlock):
        torch.nn.init.xavier_normal_(m.gamma_rh)
        torch.nn.init.zeros_(m.beta_rh)
        torch.nn.init.xavier_normal_(m.linear.weight)
        torch.nn.init.zeros_(m.linear.bias)


class AmortizedOnlineNonNegativeMatrixFactorization(NonNegativeMatrixFactorization, ValidateMixin):
    """
    Amortized version of OnlineNonNegativeMatrixFactorization.

    The idea is that OnlineNonNegativeMatrixFactorization reproduces the Kotliar cNMF results
    when using the nmf-torch-hals algorithm.
    However, this algorithm requires storage of the full loadings matrix, which is not feasible for large datasets,
    since the loadings matrix has shape (n_cells, n_components, n_replicates).
    This amortized version is a minimal change, which trains an encoder neural network to predict the loadings matrix
    from gene expression data. The trick is to handle the fact that each replicate should have an independent
    loadings matrix. Instead of training n_replicates separate encoders, we train a single encoder that takes as
    input the gene expression data and the replicate index, and uses FiLM layers to modulate activations
    based on the replicate index. This allows us to train a single encoder that can predict loadings for all
    independent replicates.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        k_values: list[int],
        r: int,
        encoder_hidden_dims: list[int],
        total_n_cells: int,
        batch_size: int,
        solver: Literal["hals", "fista"] = "fista",
        q75_convergence_threshold: float = 0.15,
        check_convergence_every_n_steps: int = 5,
        init: Literal["sklearn_random", "uniform_random"] = "uniform_random",
        transformed_data_mean: None | float = None,
    ) -> None:
        super().__init__(var_names_g=var_names_g, k_values=k_values)
        g = len(self.var_names_g)
        self.obs_names_to_index_map: dict[str, int] = {}  # used for local latents
        self.r = r
        self.solver = solver
        self.transformed_data_mean = transformed_data_mean
        # self.exponential_decay_rho = 1.0 - (batch_size / total_n_cells)  # decay factor for A and B updates, tuned
        self.exponential_decay_rho = 1.0
        self.n_batches_per_epoch = int(np.ceil(total_n_cells / batch_size))
        self.n_batches_for_forgetting_momentum = int(np.ceil(min(total_n_cells, 1e6) / batch_size))
        self.init = init
        if init == "sklearn_random":
            if transformed_data_mean is None:
                raise ValueError("transformed_data_mean must be provided when using the sklearn_random initialization")

        for i in self.k_values:
            self.register_buffer(f"A_{i}_rkk", torch.empty(r, i, i))
            self.register_buffer(f"B_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_rkg", torch.empty(r, i, g))

            # handle the encoders
            assert len(encoder_hidden_dims) > 0, "encoder_hidden_dims must be a non-empty list of hidden dimensions"
            k_encoder = ConsensusNMFEncoder(
                num_genes=g,
                hidden_dims=encoder_hidden_dims,
                num_factors=i,
                num_replicates=r,
            )
            self.add_module(f"encoder_{i}", k_encoder)

        # for training the encoder
        self.encoder_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")

        # self._D_tol = 1e-5
        self._alpha_tol = 1e-5
        # self._hals_tol = 1e-4
        # Sentinel parameter used to track the current device (mirrors OnlineNMF pattern)
        # self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self.q75_convergence_threshold = q75_convergence_threshold
        self.check_convergence_every_n_steps = check_convergence_every_n_steps
        self.convergence_tracker = NoisyConvergenceTracker(window_size=25, patience=20, min_delta=1e-4)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            m.apply(weights_init)

        match self.init:
            case "sklearn_random":
                init_fn: NMFInit = NMFInitSklearnRandom()
            case "uniform_random":
                init_fn = NMFInitUniformRandom()
            case _:
                raise ValueError(f"Unknown initialization method: {self.init}")

        for i in self.k_values:
            getattr(self, f"A_{i}_rkk").zero_()
            getattr(self, f"B_{i}_rkg").zero_()
            init_fn(getattr(self, f"D_{i}_rkg"), k=i, transformed_data_mean=self.transformed_data_mean)

        self._train_nmf_loss_ema: torch.Tensor | None = None
        self._val_nmf_loss_ema: torch.Tensor | None = None
        self.convergence_tracker.reset()

    @property
    def factors_dict(self) -> dict[int, torch.Tensor]:
        """Return the learned factors for each k value."""
        return {k: getattr(self, f"D_{k}_rkg") for k in self.k_values}

    def online_dictionary_update(self, x_ng: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.

        Args:
            x_ng: The data.
            k: The value of k to run.
            minibatch_indices_n: The indices of the cells in the current minibatch.

        Returns:
            loss: Loss for the encoder based on HALS targets.
            hals_loadings_rnk: The loadings after the HALS update, which are the targets for the encoder.
            encoder_loadings_rnk: The loadings predicted by the encoder before the update, which
        """
        # get running values
        A_rkk = getattr(self, f"A_{k}_rkk")
        B_rkg = getattr(self, f"B_{k}_rkg")
        factors_rkg = getattr(self, f"D_{k}_rkg")

        # get seed loading values from encoder (rather than from memory)
        encoder_loadings_rnk = getattr(self, f"encoder_{k}")(x_ng)

        # run nmf-torch hals online update
        if self.solver == "hals":
            solver_loadings_rnk: torch.Tensor = encoder_loadings_rnk.clone()
            updated_values = online_dictionary_update_nmf_torch_hals(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                loadings_rnk=solver_loadings_rnk,  # hals does inplace update
                A_rkk=A_rkk,
                B_rkg=B_rkg,
                n_iterations=500,
                alpha_tol=0.01,
                D_tol=0.05,
                exponential_decay_rho=self.exponential_decay_rho,
            )
        elif self.solver == "fista":
            updated_values = online_dictionary_update_fista(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                loadings_rnk=encoder_loadings_rnk.detach(),
                A_rkk=A_rkk,
                B_rkg=B_rkg,
                n_iterations=100,
                exponential_decay_rho=self.exponential_decay_rho,
            )
            solver_loadings_rnk = updated_values["loadings_rnk"]  # not inplace like hals
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        # further L1 normalize D
        D_rkg = updated_values["factors_rkg"]
        D_rkg = F.normalize(D_rkg.view(-1, D_rkg.shape[-1]), p=1, dim=-1, eps=1e-8).view_as(D_rkg)

        # update running values
        setattr(self, f"A_{k}_rkk", updated_values["A_rkk"])
        setattr(self, f"B_{k}_rkg", updated_values["B_rkg"])
        setattr(self, f"D_{k}_rkg", D_rkg)

        return {
            "loss": self.encoder_loss_fn(encoder_loadings_rnk, solver_loadings_rnk.detach()),
            "solver_loadings_rnk": solver_loadings_rnk.detach(),
            "encoder_loadings_rnk": encoder_loadings_rnk.detach(),
            "loadings_history": updated_values.get("loadings_history", None),
            "factors_history": updated_values.get("factors_history", None),
        }

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng: Gene counts matrix.
            var_names_g: The list of the variable names in the input data.
            obs_names_n: The names of the cells in the current minibatch (used when there are local latents).

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        encoder_losses = []
        nmf_reconstruction_errors = []
        for k in self.k_values:
            out = self.online_dictionary_update(x_ng=x_ng, k=k)
            encoder_loss = out["loss"]
            solver_loadings_rnk = out["solver_loadings_rnk"]
            encoder_losses.append(encoder_loss)

            # import matplotlib.pyplot as plt
            # import hashlib
            # fig = plt.figure(figsize=(10, 5))
            # plt.plot(out["loadings_history"], label="Loadings Loss")
            # plt.xlabel("Iteration")
            # plt.ylabel("Loss")
            # plt.legend()
            # plt.twinx()
            # plt.plot(out["factors_history"], label="Factors Loss", color='r')
            # md5hash_for_filename_salt = hashlib.md5(f"{torch.rand(1).item()}_{k}".encode()).hexdigest()[:12]
            # fig.savefig(f"nmf_loss_history_k{k}_{md5hash_for_filename_salt}.png")
            # plt.close(fig)

            # if we want to track the NMF loss
            with torch.no_grad():
                factors_rkg = getattr(self, f"D_{k}_rkg")
                squared_error_r = compute_reconstruction_error_compiled(
                    x_ng=x_ng,
                    loadings_rnk=solver_loadings_rnk,
                    factors_rkg=factors_rkg,
                )
                nmf_reconstruction_error = squared_error_r.mean() / (x_ng.shape[0] * x_ng.shape[1])
                nmf_reconstruction_errors.append(nmf_reconstruction_error)

        # for error computation to assess convergence
        with torch.no_grad():
            minibatch_nmf_loss = (
                sum(nmf_reconstruction_errors) / len(nmf_reconstruction_errors) if nmf_reconstruction_errors else None
            )
            beta = np.exp(-1 / self.n_batches_for_forgetting_momentum)  # momentum term for exponential moving average
            self._train_nmf_loss_ema = (
                beta * self._train_nmf_loss_ema + (1 - beta) * minibatch_nmf_loss
                if self._train_nmf_loss_ema is not None
                else minibatch_nmf_loss
            )

        return {
            "loss": sum(encoder_losses) / len(encoder_losses) if encoder_losses else None,
        }

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "OnlineNonNegativeMatrixFactorization requires that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is True, (
                "OnlineNonNegativeMatrixFactorization requires that the `broadcast_buffers` parameter of "
                "lightning.pytorch.strategies.DDPStrategy is set to True"
            )

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        if trainer.global_step % self.check_convergence_every_n_steps != 0:
            return

        beta_pow_t = np.exp(-trainer.global_step / self.n_batches_for_forgetting_momentum)

        nmf_loss_ema = self._train_nmf_loss_ema
        nmf_loss_ema_unbiased = nmf_loss_ema / (1 - beta_pow_t)
        trainer.model.log("reconstruction_error", nmf_loss_ema_unbiased, prog_bar=True)
        nmf_loss_converged = self.convergence_tracker.check_convergence(nmf_loss_ema_unbiased.item())

        # look at the actual consensus procedure histogram
        local_neighborhood_size = 0.3
        for k in self.k_values:
            D_rkg = getattr(self, f"D_{k}_rkg")
            r, num_component, g = D_rkg.shape
            d_norm_rkg = F.normalize(D_rkg, dim=-1, p=2)
            d_norm_mg = d_norm_rkg.reshape(r * num_component, g)

            if r > 1:
                n_neighbors = int(r * local_neighborhood_size)
                if n_neighbors < 2:
                    warnings.warn(
                        f"during convergence check, "
                        f"local_neighborhood_size {local_neighborhood_size} is too small for k={num_component}. "
                        f"n_neighbors = int(replicates * local_neighborhood_size) = {n_neighbors}. "
                        "We want n_neighbors >= 2. Increase local_neighborhood_size."
                    )

                # euclidean distance to every other run
                euclidean_dist_mm = torch.cdist(d_norm_mg, d_norm_mg, p=2)
                euclidean_dist_mm.fill_diagonal_(0)  # correct for roundoff errors that may be present

                # top n_neighbors plus self (distance 0)
                n_nearest_dist_including_self_mL, _ = torch.topk(euclidean_dist_mm, n_neighbors + 1, largest=False)

                # distances to top n_neighbors
                n_nearest_dist_ml = n_nearest_dist_including_self_mL[:, 1:]

                # mean distance to top n_neighbors
                mean_neighbor_distance_m = n_nearest_dist_ml.mean(dim=1)

                # log historgram
                for logger in trainer.loggers:
                    if isinstance(logger, pl.loggers.TensorBoardLogger):
                        logger.experiment.add_histogram(
                            f"k={k}__consensus_histogram",
                            mean_neighbor_distance_m,
                            global_step=trainer.global_step,
                            bins=np.linspace(0, 1, 75),
                        )

                trainer.model.log(f"k={k}__consensus_L1", mean_neighbor_distance_m.mean(), prog_bar=False)
                # trainer.model.log(f"k={k}__consensus_L2", mean_neighbor_distance_m.pow(2).mean(), prog_bar=False)
                trainer.model.log(f"k={k}__consensus_q75", mean_neighbor_distance_m.quantile(0.75), prog_bar=True)

        # implement convergence check: reconstruction error plateaued and consensus_q75 is below the threshold
        # at least one epoch
        # do not stop near an A, B reset
        # look for loss convergence
        # and look for consensus convergence (q75 of mean distance to neighbors below threshold)
        converged = (
            (trainer.global_step > self.n_batches_for_forgetting_momentum)
            and (
                trainer.global_step % self.n_batches_for_forgetting_momentum
                >= min(100, self.n_batches_for_forgetting_momentum * 0.75)
            )
            and nmf_loss_converged
            and (
                np.percentile(
                    [trainer.callback_metrics.get(f"k={k}__consensus_q75", float("inf")) for k in self.k_values], q=10
                )
                <= self.q75_convergence_threshold
            )
        )
        if converged:
            trainer.should_stop = True
            print("Stopping early: converged")

        if trainer.global_step % self.n_batches_for_forgetting_momentum == 0:
            # this hard reset to zero is equivalent to forgetting momentum
            for i in self.k_values:
                getattr(self, f"A_{i}_rkk").zero_()
                getattr(self, f"B_{i}_rkg").zero_()

    def on_end(self, trainer: pl.Trainer) -> None:
        trainer.save_checkpoint(trainer.default_root_dir + "/NMF.ckpt")

    @torch.no_grad()
    def infer_loadings(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        consensus_factors: dict[int, dict[str, torch.Tensor | float]],
        k: int,
        normalize: bool = False,
        obs_names_n: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Infer the loadings of each program for the input count matrix.
        To be run after the model has been trained.
        """
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        D_kg = consensus_factors[k]["consensus_D_kg"]
        assert isinstance(D_kg, torch.Tensor), "consensus_D_kg must be a tensor"

        alpha_nk = (
            solve_nnls_fista(  # TODO cehck if we can use other fista solver
                D_kg.to(x_ng.device).unsqueeze(0).transpose(1, 2),
                x_ng.t(),
                tol=self._alpha_tol * 0.1,
                max_iter=1000,
            )
            .transpose(1, 2)
            .squeeze(0)
        )

        if normalize:
            alpha_nk = F.normalize(alpha_nk, p=1, dim=-1)

        return alpha_nk

    def validate(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
    ) -> None:
        """
        Args:
            x_ng: Gene counts matrix.
            var_names_g: The list of the variable names in the input data.

        Returns:
            An empty dictionary.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        nmf_reconstruction_errors = []
        for k in self.k_values:
            encoder_loadings_rnk = getattr(self, f"encoder_{k}")(x_ng)
            factors_rkg = getattr(self, f"D_{k}_rkg")
            squared_error_r = compute_reconstruction_error_compiled(
                x_ng=x_ng,
                loadings_rnk=encoder_loadings_rnk,
                factors_rkg=factors_rkg,
            )
            nmf_reconstruction_error = squared_error_r.mean() / (x_ng.shape[0] * x_ng.shape[1])
            nmf_reconstruction_errors.append(nmf_reconstruction_error)

        # for error computation to assess convergence
        minibatch_nmf_loss = (
            sum(nmf_reconstruction_errors) / len(nmf_reconstruction_errors) if nmf_reconstruction_errors else None
        )
        beta = np.exp(-1 / self.n_batches_for_forgetting_momentum)  # momentum term for exponential moving average
        self._val_nmf_loss_ema = (
            beta * self._val_nmf_loss_ema + (1 - beta) * minibatch_nmf_loss
            if self._val_nmf_loss_ema is not None
            else minibatch_nmf_loss
        )

        # Logging to TensorBoard by default
        pl_module.log("val_nmf_loss", self._val_nmf_loss_ema, sync_dist=True, on_epoch=True)

    @torch.no_grad()
    def reconstruction_error(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        consensus_factors: dict[int, dict[str, torch.Tensor | float]],
    ) -> dict[int, float]:
        """
        Compute the reconstruction error for each k_value using trained consensus factors D_kg.

        Args:
            x_ng: Gene counts matrix.
            var_names_g: The list of the variable names in the input data.
            consensus_factors: The consensus factors for each k_value are in consensus_factors[k]["consensus_D_kg"].

        Returns:
            A dictionary mapping each k_value to its reconstruction error.
        """
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        rec_error = {}
        for k in consensus_factors.keys():
            D_kg = consensus_factors[k]["consensus_D_kg"]
            assert isinstance(D_kg, torch.Tensor), "consensus_D_kg must be a tensor"
            if (D_kg == 0).all():
                raise ValueError("D_kg is all zeros, please train the model and run compute_consensus_factors() first")

            alpha_nk = self.infer_loadings(
                x_ng=x_ng,
                var_names_g=var_names_g,
                consensus_factors=consensus_factors,
                k=k,
                normalize=False,
            ).squeeze(0)

            rec_error[k] = (
                nmf_frobenius_loss(
                    x_ng=x_ng,
                    loadings_nk=alpha_nk.to(x_ng.device),
                    factors_kg=D_kg.to(x_ng.device),
                )
                .sum()
                .item()
            )

        return rec_error
