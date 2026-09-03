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
    frobenius_loss_trace_compiled,
    nmf_frobenius_loss,
    online_dictionary_update_fista,
    online_dictionary_update_nmf_torch_hals,
    solve_nnls_fista,
)
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class BilinearLoadingsEncoder(torch.nn.Module):
    """
    Encoder that warm-starts NMF loadings from gene expression data and the current factors.

    Replicates are independent by construction: the factor encoder processes each replicate's
    factors as an independent batch dimension, and there is no cross-replicate attention.

    The bilinear affinity c_ne · f_rke^T is mechanistically motivated: it measures how much
    each cell (in latent space) resembles each factor, which is the gradient signal FISTA uses
    on its first step. Applying log1p to the cell input compresses count-data heavy tails while
    keeping factors in their natural L1-normalized probability scale.
    """

    def __init__(self, n_genes: int, latent_dim: int):
        super().__init__()
        self.cell_encoder = torch.nn.Linear(n_genes, latent_dim, bias=False)
        self.factor_encoder = torch.nn.Linear(n_genes, latent_dim, bias=False)
        self.scale = latent_dim**-0.5

    def forward(self, x_ng: torch.Tensor, w_rkg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ng: Gene counts of shape (N, G).
            w_rkg: Current NMF factors of shape (R, K, G), L1-normalized by row.
        Returns:
            h_rnk: Warm-start loadings of shape (R, N, K), non-negative, scaled so each
                cell's loadings sum approximately to its total count.
        """
        c_ne = self.cell_encoder(torch.log1p(x_ng))
        f_rke = self.factor_encoder(w_rkg)
        logits_rnk = torch.einsum("ne,rke->rnk", c_ne, f_rke) * self.scale
        h_sparse_rnk = F.relu(logits_rnk)
        h_norm_rnk = h_sparse_rnk / h_sparse_rnk.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return h_norm_rnk * x_ng.sum(dim=-1, keepdim=True).unsqueeze(0)


def weights_init(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class AmortizedOnlineNonNegativeMatrixFactorization(NonNegativeMatrixFactorization, ValidateMixin):
    """
    Amortized version of OnlineNonNegativeMatrixFactorization.

    The idea is that OnlineNonNegativeMatrixFactorization reproduces the Kotliar cNMF results
    when using the nmf-torch-hals algorithm.
    However, this algorithm requires storage of the full loadings matrix, which is not feasible for large datasets,
    since the loadings matrix has shape (n_cells, n_components, n_replicates).
    This amortized version is a minimal change, which trains an encoder neural network to predict the loadings matrix
    from gene expression data. The encoder conditions on both the cell expression and the current factor matrix,
    using a bilinear dot product to compute loadings affinities. Replicates are kept independent by construction
    since each replicate's factors are processed as an independent batch dimension with no cross-replicate attention.

    Stopping uses a dual-signal AND criterion based on two independent per-k signals: the encoder loss and the
    NMF reconstruction loss. Both are measured as trailing-window averages over the last
    ``trailing_window_fraction`` of each monitoring period, which avoids contaminating estimates with the
    transient spike that follows each catastrophic-forgetting reset. The per-step relative improvement rate is
    computed for each signal independently, with separate patience counters (``_encoder_trigger_patience_counters``
    and ``_recon_trigger_patience_counters``). Cooldown begins only when **both** signals have failed to improve
    for ``trigger_patience`` consecutive periods — whichever signal is slower is the trailing condition.
    Cooldown (A/B accumulation without forgetting) ends when **both** signals again plateau for
    ``cooldown_patience`` consecutive periods, or when ``max_cooldown_epochs`` elapses, whichever comes first.
    Thresholds are independently tunable via ``encoder_improvement_threshold`` and
    ``reconstruction_improvement_threshold``.
    """

    def __init__(
        self,
        var_names_g: Sequence[str],
        k_values: list[int],
        r: int,
        latent_dim: int,
        total_n_cells: int,
        batch_size: int,
        solver: Literal["hals", "fista"] = "fista",
        encoder_improvement_threshold: float = 2e-3,
        reconstruction_improvement_threshold: float = 2e-3,
        trigger_patience: int = 3,
        cooldown_patience: int = 5,
        max_cooldown_epochs: float = 1.0,
        trailing_window_fraction: float = 0.25,
        exploration_epochs: int = 2,
        max_solver_iter_train: int = 50,
        max_solver_iter_cooldown: int = 200,
        init: Literal["sklearn_random", "uniform_random"] = "uniform_random",
        transformed_data_mean: None | float = None,
    ) -> None:
        super().__init__(var_names_g=var_names_g, k_values=k_values)
        g = len(self.var_names_g)
        self.obs_names_to_index_map: dict[str, int] = {}  # used for local latents
        self.r = r
        self.solver = solver
        self.transformed_data_mean = transformed_data_mean
        self.exponential_decay_rho = 1.0
        self.n_batches_per_epoch = int(np.ceil(total_n_cells / batch_size))
        self.n_batches_for_forgetting_momentum = int(np.ceil(min(total_n_cells, 1e6) / batch_size))
        self.trailing_window_size = max(5, int(self.n_batches_for_forgetting_momentum * trailing_window_fraction))
        self.init = init
        if init == "sklearn_random":
            if transformed_data_mean is None:
                raise ValueError("transformed_data_mean must be provided when using the sklearn_random initialization")

        self.encoder = BilinearLoadingsEncoder(n_genes=g, latent_dim=latent_dim)

        for i in self.k_values:
            self.register_buffer(f"A_{i}_rkk", torch.empty(r, i, i))
            self.register_buffer(f"B_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_rkg", torch.empty(r, i, g))

        # for training the encoder
        self.encoder_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")

        self._alpha_tol = 1e-5
        self.encoder_improvement_threshold = encoder_improvement_threshold
        self.reconstruction_improvement_threshold = reconstruction_improvement_threshold
        self.trigger_patience = trigger_patience
        self.cooldown_patience = cooldown_patience
        self.max_cooldown_epochs = max_cooldown_epochs
        self.exploration_epochs = exploration_epochs
        self.max_solver_iter_train = max_solver_iter_train
        self.max_solver_iter_cooldown = max_solver_iter_cooldown
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
        self._D_prev_snapshots: dict[int, torch.Tensor | None] = {k: None for k in self.k_values}
        self._last_encoder_loss: dict[int, float | None] = {k: None for k in self.k_values}
        self._encoder_loss_window_sum: dict[int, float] = {k: 0.0 for k in self.k_values}
        self._encoder_loss_window_count: dict[int, int] = {k: 0 for k in self.k_values}
        self._encoder_loss_snapshot: dict[int, float | None] = {k: None for k in self.k_values}
        self._last_recon_loss: dict[int, float | None] = {k: None for k in self.k_values}
        self._recon_loss_window_sum: dict[int, float] = {k: 0.0 for k in self.k_values}
        self._recon_loss_window_count: dict[int, int] = {k: 0 for k in self.k_values}
        self._recon_loss_snapshot: dict[int, float | None] = {k: None for k in self.k_values}
        self._encoder_trigger_patience_counters: dict[int, int] = {k: 0 for k in self.k_values}
        self._recon_trigger_patience_counters: dict[int, int] = {k: 0 for k in self.k_values}
        self._encoder_cooldown_patience_counters: dict[int, int] = {k: 0 for k in self.k_values}
        self._recon_cooldown_patience_counters: dict[int, int] = {k: 0 for k in self.k_values}
        self._k_in_final_epoch: dict[int, bool] = {k: False for k in self.k_values}
        self._k_final_epoch_start: dict[int, int | None] = {k: None for k in self.k_values}
        self._k_done: dict[int, bool] = {k: False for k in self.k_values}

    @property
    def factors_dict(self) -> dict[int, torch.Tensor]:
        """Return the learned factors for each k value."""
        return {k: getattr(self, f"D_{k}_rkg") for k in self.k_values}

    def online_dictionary_update(self, x_ng: torch.Tensor, k: int, n_iterations: int = 100) -> dict[str, torch.Tensor]:
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
        encoder_loadings_rnk = self.encoder(x_ng, factors_rkg.detach())

        # run nmf-torch hals online update
        if self.solver == "hals":
            solver_loadings_rnk: torch.Tensor = encoder_loadings_rnk.clone()
            updated_values = online_dictionary_update_nmf_torch_hals(
                x_ng=x_ng,
                factors_rkg=factors_rkg,
                loadings_rnk=solver_loadings_rnk,  # hals does inplace update
                A_rkk=A_rkk,
                B_rkg=B_rkg,
                n_iterations=n_iterations,
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
                n_iterations=n_iterations,
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

        encoder_losses: list[torch.Tensor] = []
        nmf_reconstruction_errors: list[torch.Tensor] = []
        for k in self.k_values:
            if self._k_done[k]:
                continue
            n_iter = self.max_solver_iter_cooldown if self._k_in_final_epoch[k] else self.max_solver_iter_train
            out = self.online_dictionary_update(x_ng=x_ng, k=k, n_iterations=n_iter)
            encoder_loss = out["loss"]
            solver_loadings_rnk = out["solver_loadings_rnk"]
            encoder_losses.append(encoder_loss)

            with torch.no_grad():
                # store latest encoder loss for trailing-window accumulation in on_train_batch_end
                self._last_encoder_loss[k] = encoder_loss.detach().item()

                # track NMF reconstruction error
                factors_rkg = getattr(self, f"D_{k}_rkg")
                squared_error_r = frobenius_loss_trace_compiled(
                    x_ng=x_ng,
                    h_rnk=solver_loadings_rnk,
                    w_rkg=factors_rkg,
                )
                nmf_reconstruction_error = squared_error_r.mean() / (x_ng.shape[0] * x_ng.shape[1])
                nmf_reconstruction_errors.append(nmf_reconstruction_error)
                self._last_recon_loss[k] = nmf_reconstruction_error.item()

        # for error computation to assess convergence
        with torch.no_grad():
            minibatch_nmf_loss = (
                sum(nmf_reconstruction_errors) / len(nmf_reconstruction_errors) if nmf_reconstruction_errors else None
            )
            beta = np.exp(-1 / self.n_batches_for_forgetting_momentum)  # momentum term for exponential moving average
            val = (
                beta * self._train_nmf_loss_ema + (1 - beta) * minibatch_nmf_loss
                if self._train_nmf_loss_ema is not None
                else minibatch_nmf_loss
            )
            assert isinstance(val, torch.Tensor)
            self._train_nmf_loss_ema = val

        loss = sum(encoder_losses) / len(encoder_losses) if encoder_losses else None
        assert isinstance(loss, torch.Tensor) or loss is None

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
        step = trainer.global_step
        if step == 0:
            return

        step_within_period = step % self.n_batches_for_forgetting_momentum
        in_trailing_window = step_within_period >= self.n_batches_for_forgetting_momentum - self.trailing_window_size
        at_period_boundary = step_within_period == 0

        if in_trailing_window:
            for k in self.k_values:
                if self._k_done[k]:
                    continue
                if self._last_encoder_loss[k] is not None:
                    local_loss = self._last_encoder_loss[k]
                    assert isinstance(local_loss, float)
                    self._encoder_loss_window_sum[k] += local_loss
                    self._encoder_loss_window_count[k] += 1
                if self._last_recon_loss[k] is not None:
                    local_recon_loss = self._last_recon_loss[k]
                    assert isinstance(local_recon_loss, float)
                    self._recon_loss_window_sum[k] += local_recon_loss
                    self._recon_loss_window_count[k] += 1

        if not at_period_boundary:
            return

        module = trainer.model
        assert isinstance(module, pl.LightningModule)

        # --- log reconstruction error EMA ---
        if self._train_nmf_loss_ema is not None:
            beta_pow_t = np.exp(-step / self.n_batches_for_forgetting_momentum)
            nmf_loss_ema_unbiased = self._train_nmf_loss_ema / (1 - beta_pow_t)
            module.log("rec_error", nmf_loss_ema_unbiased, prog_bar=True)

        # --- log consensus metrics ---
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

                euclidean_dist_mm = torch.cdist(d_norm_mg, d_norm_mg, p=2)
                euclidean_dist_mm.fill_diagonal_(0)
                n_nearest_dist_including_self_mL, _ = torch.topk(euclidean_dist_mm, n_neighbors + 1, largest=False)
                n_nearest_dist_ml = n_nearest_dist_including_self_mL[:, 1:]
                mean_neighbor_distance_m = n_nearest_dist_ml.mean(dim=1)

                for logger in trainer.loggers:
                    try:
                        if isinstance(logger, pl.loggers.TensorBoardLogger):
                            logger.experiment.add_histogram(
                                f"k={k}_consensus_histogram",
                                mean_neighbor_distance_m,
                                global_step=step,
                                bins=np.linspace(0, 1, 75),
                            )
                    except Exception as e:
                        warnings.warn(f"Failed to log histogram for k={k} step={step} due to {e}")

                module.log(f"k={k}_consensus_L1", mean_neighbor_distance_m.mean(), prog_bar=False)
                module.log(f"k={k}_consensus_q75", mean_neighbor_distance_m.quantile(0.75), prog_bar=True)

        # --- per-k dual-signal AND stopping and forgetting ---
        min_steps_before_stop = self.exploration_epochs * self.n_batches_per_epoch
        # Maximum cooldown: at least max_cooldown_epochs full data passes, but never fewer steps
        # than cooldown_patience monitoring periods (so the patience counter always has room to fire).
        max_cooldown_steps = max(
            int(self.max_cooldown_epochs * self.n_batches_per_epoch),
            self.cooldown_patience * self.n_batches_for_forgetting_momentum,
        )

        all_done = all(self._k_done[k] for k in self.k_values)
        for k in self.k_values:
            if self._k_done[k]:
                continue

            # compute and log drift for monitoring (decision-free)
            D_rkg = getattr(self, f"D_{k}_rkg").detach()
            prev_D = self._D_prev_snapshots[k]
            if prev_D is not None:
                prev_norm = prev_D.norm()
                drift = ((D_rkg - prev_D).norm() / prev_norm).item() if prev_norm > 1e-8 else float("inf")
                module.log(f"k={k}__forgetting_drift", drift, prog_bar=False)

            # --- encoder loss: compute trailing-window average and improvement rate ---
            enc_window_count = self._encoder_loss_window_count[k]
            current_encoder_loss = self._encoder_loss_window_sum[k] / enc_window_count if enc_window_count > 0 else None
            self._encoder_loss_window_sum[k] = 0.0
            self._encoder_loss_window_count[k] = 0

            # --- reconstruction loss: compute trailing-window average and improvement rate ---
            recon_window_count = self._recon_loss_window_count[k]
            current_recon_loss = self._recon_loss_window_sum[k] / recon_window_count if recon_window_count > 0 else None
            self._recon_loss_window_sum[k] = 0.0
            self._recon_loss_window_count[k] = 0

            # log per-k reconstruction loss (raw window average)
            if current_recon_loss is not None:
                module.log(f"k={k}_rec_error", current_recon_loss, prog_bar=False)

            encoder_rate: float | None = None
            recon_rate: float | None = None

            if current_encoder_loss is not None:
                prev_encoder_loss = self._encoder_loss_snapshot[k]
                if prev_encoder_loss is not None and prev_encoder_loss > 1e-8 and step >= min_steps_before_stop:
                    encoder_rate = (prev_encoder_loss - current_encoder_loss) / (
                        prev_encoder_loss * self.n_batches_for_forgetting_momentum
                    )
                    module.log(f"k={k}_encoder_improvement_rate", encoder_rate, prog_bar=False)
                self._encoder_loss_snapshot[k] = current_encoder_loss

            if current_recon_loss is not None:
                prev_recon_loss = self._recon_loss_snapshot[k]
                if prev_recon_loss is not None and prev_recon_loss > 1e-8 and step >= min_steps_before_stop:
                    recon_rate = (prev_recon_loss - current_recon_loss) / (
                        prev_recon_loss * self.n_batches_for_forgetting_momentum
                    )
                    module.log(f"k={k}_recon_improvement_rate", recon_rate, prog_bar=False)
                self._recon_loss_snapshot[k] = current_recon_loss

            if encoder_rate is not None and recon_rate is not None:
                if not self._k_in_final_epoch[k]:
                    # exploration phase: update independent patience counters for each signal
                    if encoder_rate < self.encoder_improvement_threshold:
                        self._encoder_trigger_patience_counters[k] += 1
                    else:
                        self._encoder_trigger_patience_counters[k] = 0

                    if recon_rate < self.reconstruction_improvement_threshold:
                        self._recon_trigger_patience_counters[k] += 1
                    else:
                        self._recon_trigger_patience_counters[k] = 0

                    # AND: both signals must reach patience before entering cooldown
                    if (
                        self._encoder_trigger_patience_counters[k] >= self.trigger_patience
                        and self._recon_trigger_patience_counters[k] >= self.trigger_patience
                    ):
                        self._k_in_final_epoch[k] = True
                        self._k_final_epoch_start[k] = step
                        print(f"Triggering cooldown for k={k} at step={step}")
                else:
                    # cooldown phase: check hard cap first
                    k_start = self._k_final_epoch_start[k]
                    assert k_start is not None
                    if step - k_start >= max_cooldown_steps:
                        self._k_done[k] = True
                        continue

                    # update independent patience counters for each signal
                    if encoder_rate < self.encoder_improvement_threshold:
                        self._encoder_cooldown_patience_counters[k] += 1
                    else:
                        self._encoder_cooldown_patience_counters[k] = 0

                    if recon_rate < self.reconstruction_improvement_threshold:
                        self._recon_cooldown_patience_counters[k] += 1
                    else:
                        self._recon_cooldown_patience_counters[k] = 0

                    # AND: both signals must reach patience before marking done
                    if (
                        self._encoder_cooldown_patience_counters[k] >= self.cooldown_patience
                        and self._recon_cooldown_patience_counters[k] >= self.cooldown_patience
                    ):
                        self._k_done[k] = True
                        continue

            if not self._k_in_final_epoch[k]:
                # snapshot D then forget; log the forgetting event
                self._D_prev_snapshots[k] = D_rkg.clone()
                getattr(self, f"A_{k}_rkk").zero_()
                getattr(self, f"B_{k}_rkg").zero_()
                module.log(f"k={k}_forgetting", 1.0, prog_bar=False)
            else:
                module.log(f"k={k}_forgetting", 0.0, prog_bar=False)
            # during cooldown: do not forget; A/B accumulate

        all_done = all(self._k_done[k] for k in self.k_values)
        n_k_still_training = sum(not self._k_done[k] for k in self.k_values)
        module.log("k_training", float(n_k_still_training), prog_bar=True)

        if all_done:
            trainer.should_stop = True
            print("Stopping early: all k values have completed cooldown")

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
            factors_rkg = getattr(self, f"D_{k}_rkg")
            encoder_loadings_rnk = self.encoder(x_ng, factors_rkg.detach())
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
        val = (
            beta * self._val_nmf_loss_ema + (1 - beta) * minibatch_nmf_loss
            if self._val_nmf_loss_ema is not None
            else minibatch_nmf_loss
        )
        assert isinstance(val, torch.Tensor) or val is None
        self._val_nmf_loss_ema = val

        # Logging to TensorBoard by default
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
