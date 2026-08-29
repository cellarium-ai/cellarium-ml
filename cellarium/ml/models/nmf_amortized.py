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

    def forward(self, x: torch.Tensor, active_replicates: list[int] | None = None) -> torch.Tensor:
        # linear layer
        h = self.linear(x)

        # batch norm
        if h.dim() == 3:
            # collapse first two dimensions r and n to apply batch norm, then un-collapse
            r, n, h_dim = h.shape
            h_rnh = self.batch_norm(h.view(r * n, h_dim)).view(r, n, h_dim)
        else:
            h_rnh = self.batch_norm(h).unsqueeze(0)  # expand to (1, N, H)

        # index FiLM params for active replicates only, then broadcast across n
        if active_replicates is not None:
            gamma_r1h = self.gamma_rh[active_replicates].unsqueeze(1)
            beta_r1h = self.beta_rh[active_replicates].unsqueeze(1)
        else:
            gamma_r1h = self.gamma_rh.unsqueeze(1)
            beta_r1h = self.beta_rh.unsqueeze(1)

        return self.relu(gamma_r1h * h_rnh + beta_r1h)


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
        self.output_layer = torch.nn.Linear(prev_dim, num_factors)

        # NMF loadings must be non-negative
        self.relu = torch.nn.ReLU()

    def forward(self, x_ng: torch.Tensor, active_replicates: list[int] | None = None) -> torch.Tensor:
        """
        Args:
            x_ng: Input gene expression of shape (N, G)
            active_replicates: Replicate indices to compute. If None, all replicates are used.
        Returns:
            loadings_rnk: NMF loadings of shape (R_active, N, K)
        """
        h = x_ng
        for block in self.blocks:
            h = block(h, active_replicates=active_replicates)
        return self.relu(self.output_layer(h))


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

    Convergence is assessed per-replicate per-k using factor drift: the relative Frobenius norm of change
    in D over a fixed window of steps. A replicate is converged when it has been below `factor_drift_threshold`
    for `convergence_patience` consecutive windows. The window size is derived automatically from the forgetting
    period. Convergence state is reset whenever the A/B forgetting step fires, ensuring that post-reset
    exploration is allowed to re-settle before stopping.
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
        factor_drift_threshold: float = 0.01,
        convergence_patience: int = 3,
        init: Literal["sklearn_random", "uniform_random"] = "uniform_random",
        transformed_data_mean: None | float = None,
    ) -> None:
        super().__init__(var_names_g=var_names_g, k_values=k_values)
        g = len(self.var_names_g)
        self.obs_names_to_index_map: dict[str, int] = {}
        self.r = r
        self.solver = solver
        self.transformed_data_mean = transformed_data_mean
        self.exponential_decay_rho = 1.0
        self.n_batches_per_epoch = int(np.ceil(total_n_cells / batch_size))
        self.n_batches_for_forgetting_momentum = int(np.ceil(min(total_n_cells, 1e6) / batch_size))
        # Window size for convergence checks: fixed fraction of the forgetting period.
        # The drift threshold means "fractional change over convergence_window_size steps < threshold",
        # which is stable regardless of how frequently the user inspects logs.
        self.convergence_window_size = max(1, self.n_batches_for_forgetting_momentum // 20)
        self.factor_drift_threshold = factor_drift_threshold
        self.convergence_patience = convergence_patience
        self.init = init
        self.k_to_idx: dict[int, int] = {k: i for i, k in enumerate(k_values)}

        if init == "sklearn_random":
            if transformed_data_mean is None:
                raise ValueError("transformed_data_mean must be provided when using the sklearn_random initialization")

        if convergence_patience * self.convergence_window_size >= self.n_batches_for_forgetting_momentum:
            warnings.warn(
                f"convergence_patience * convergence_window_size "
                f"({convergence_patience} * {self.convergence_window_size} = "
                f"{convergence_patience * self.convergence_window_size}) >= "
                f"n_batches_for_forgetting_momentum ({self.n_batches_for_forgetting_momentum}). "
                "Convergence may never be declared — consider reducing convergence_patience."
            )

        for i in self.k_values:
            self.register_buffer(f"A_{i}_rkk", torch.empty(r, i, i))
            self.register_buffer(f"B_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_rkg", torch.empty(r, i, g))

            assert len(encoder_hidden_dims) > 0, "encoder_hidden_dims must be a non-empty list of hidden dimensions"
            self.add_module(
                f"encoder_{i}",
                ConsensusNMFEncoder(num_genes=g, hidden_dims=encoder_hidden_dims, num_factors=i, num_replicates=r),
            )

        self.encoder_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self._alpha_tol = 1e-5

        # Per-replicate-per-k convergence state (shape: R x len(k_values))
        self.register_buffer("converged_rK", torch.zeros(r, len(k_values), dtype=torch.bool))
        self.register_buffer("patience_counter_rK", torch.zeros(r, len(k_values), dtype=torch.long))

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

        # self._train_nmf_loss_ema: torch.Tensor | None = None
        self._val_nmf_loss_ema: torch.Tensor | None = None
        self._D_snapshots: dict[int, torch.Tensor | None] = {k: None for k in self.k_values}
        self.converged_rK.zero_()
        self.patience_counter_rK.zero_()

    @torch.no_grad()
    def _compute_factor_drift(self, D_kg: torch.Tensor, D_kg_prev: torch.Tensor) -> float:
        """Relative Frobenius norm of change in factors for a single replicate."""
        denom = D_kg_prev.norm()
        if denom < 1e-8:
            return float("inf")
        return ((D_kg - D_kg_prev).norm() / denom).item()

    @property
    def factors_dict(self) -> dict[int, torch.Tensor]:
        """Return the learned factors for each k value."""
        return {k: getattr(self, f"D_{k}_rkg") for k in self.k_values}

    def online_dictionary_update(
        self, x_ng: torch.Tensor, k: int, active_replicate_indices: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.
        Only processes the specified active replicates; converged replicate rows in D/A/B are untouched.
        """
        # Slice buffers to active replicates only
        A_rkk = getattr(self, f"A_{k}_rkk")[active_replicate_indices]
        B_rkg = getattr(self, f"B_{k}_rkg")[active_replicate_indices]
        factors_rkg = getattr(self, f"D_{k}_rkg")[active_replicate_indices]

        # Encoder warm-start for active replicates only
        encoder_loadings_rnk = getattr(self, f"encoder_{k}")(x_ng, active_replicates=active_replicate_indices)

        if self.solver == "hals":
            solver_loadings_rnk: torch.Tensor = encoder_loadings_rnk.clone()
            updated_values = online_dictionary_update_nmf_torch_hals(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                loadings_rnk=solver_loadings_rnk,
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
            solver_loadings_rnk = updated_values["loadings_rnk"]
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        # L1-normalize D for active replicates
        D_active = updated_values["factors_rkg"]
        D_active = F.normalize(D_active.view(-1, D_active.shape[-1]), p=1, dim=-1, eps=1e-8).view_as(D_active)

        # Write back only the active replicate rows (in-place index assignment keeps converged rows frozen)
        getattr(self, f"A_{k}_rkk")[active_replicate_indices] = updated_values["A_rkk"]
        getattr(self, f"B_{k}_rkg")[active_replicate_indices] = updated_values["B_rkg"]
        getattr(self, f"D_{k}_rkg")[active_replicate_indices] = D_active

        return {
            "loss": self.encoder_loss_fn(encoder_loadings_rnk, solver_loadings_rnk.detach()),
            "solver_loadings_rnk": solver_loadings_rnk.detach(),
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

        Returns:
            A dict with the encoder loss (averaged over active k values and replicates).
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        encoder_losses: list[torch.Tensor] = []
        # nmf_reconstruction_errors = []
        for k in self.k_values:
            k_idx = self.k_to_idx[k]
            active_indices = (~self.converged_rK[:, k_idx]).nonzero(as_tuple=True)[0].tolist()
            if not active_indices:
                continue

            out = self.online_dictionary_update(x_ng=x_ng, k=k, active_replicate_indices=active_indices)
            encoder_losses.append(out["loss"])

        #     with torch.no_grad():
        #         squared_error_r = compute_reconstruction_error_compiled(
        #             x_ng=x_ng,
        #             loadings_rnk=out["solver_loadings_rnk"],
        #             factors_rkg=getattr(self, f"D_{k}_rkg")[active_indices],
        #         )
        #         nmf_reconstruction_errors.append(squared_error_r.mean() / (x_ng.shape[0] * x_ng.shape[1]))

        # with torch.no_grad():
        #     if nmf_reconstruction_errors:
        #         minibatch_nmf_loss = sum(nmf_reconstruction_errors) / len(nmf_reconstruction_errors)
        #         assert isinstance(minibatch_nmf_loss, torch.Tensor)
        #         beta = np.exp(-1 / self.n_batches_for_forgetting_momentum)
        #         self._train_nmf_loss_ema = (
        #             beta * self._train_nmf_loss_ema + (1 - beta) * minibatch_nmf_loss
        #             if self._train_nmf_loss_ema is not None
        #             else minibatch_nmf_loss
        #         )

        loss = sum(encoder_losses) / len(encoder_losses) if encoder_losses else None
        assert loss is None or isinstance(loss, torch.Tensor)
        return {"loss": loss}

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
        # # Log reconstruction error EMA every step
        # if self._train_nmf_loss_ema is not None:
        #     beta_pow_t = np.exp(-trainer.global_step / self.n_batches_for_forgetting_momentum)
        #     assert isinstance(trainer.model, pl.LightningModule)
        #     trainer.model.log("reconstruction_error", self._train_nmf_loss_ema / (1 - beta_pow_t), prog_bar=True)

        # Forgetting reset: zero A and B, then reset convergence state so replicates must
        # re-demonstrate stability after the exploration phase.
        if trainer.global_step % self.n_batches_for_forgetting_momentum == 0:
            for i in self.k_values:
                getattr(self, f"A_{i}_rkk").zero_()
                getattr(self, f"B_{i}_rkg").zero_()
            self._D_snapshots = {k: None for k in self.k_values}
            self.patience_counter_rK.zero_()
            return

        # Convergence check every convergence_window_size steps
        if trainer.global_step % self.convergence_window_size != 0:
            return

        n_active_total = 0

        for k in self.k_values:
            k_idx = self.k_to_idx[k]
            D_full = getattr(self, f"D_{k}_rkg").detach()

            if self._D_snapshots[k] is None:
                # First window after a reset: just store the baseline snapshot
                self._D_snapshots[k] = D_full.clone()
                continue

            snapshot = self._D_snapshots[k]
            assert isinstance(snapshot, torch.Tensor)
            for r in range(self.r):
                if self.converged_rK[r, k_idx]:
                    continue
                drift = self._compute_factor_drift(D_full[r], snapshot[r])
                if drift < self.factor_drift_threshold:
                    self.patience_counter_rK[r, k_idx] += 1
                    if self.patience_counter_rK[r, k_idx] >= self.convergence_patience:
                        self.converged_rK[r, k_idx] = True
                else:
                    self.patience_counter_rK[r, k_idx] = 0

            self._D_snapshots[k] = D_full.clone()

            n_active = int((~self.converged_rK[:, k_idx]).sum())
            n_active_total += n_active
            assert isinstance(trainer.model, pl.LightningModule)
            trainer.model.log(f"k={k}__n_active_replicates", float(n_active), prog_bar=False)

        assert isinstance(trainer.model, pl.LightningModule)
        trainer.model.log("n_training", float(n_active_total), prog_bar=True)

        if self.converged_rK.all():
            trainer.should_stop = True
            print("Stopping early: all replicates converged")

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
            solve_nnls_fista(
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
            nmf_reconstruction_errors.append(squared_error_r.mean() / (x_ng.shape[0] * x_ng.shape[1]))

        minibatch_nmf_loss = (
            sum(nmf_reconstruction_errors) / len(nmf_reconstruction_errors) if nmf_reconstruction_errors else None
        )
        beta = np.exp(-1 / self.n_batches_for_forgetting_momentum)
        vloss = (
            beta * self._val_nmf_loss_ema + (1 - beta) * minibatch_nmf_loss
            if self._val_nmf_loss_ema is not None
            else minibatch_nmf_loss
        )
        assert vloss is None or isinstance(vloss, torch.Tensor)
        self._val_nmf_loss_ema = vloss
        if self._val_nmf_loss_ema is not None:
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
