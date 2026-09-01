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
        latent_dim: int,
        total_n_cells: int,
        batch_size: int,
        solver: Literal["hals", "fista"] = "fista",
        forgetting_drift_threshold: float = 0.1,
        forgetting_patience: int = 5,
        exploration_epochs: int = 2,
        cooldown_periods: int | None = None,
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

            self.add_module(f"encoder_{i}", BilinearLoadingsEncoder(n_genes=g, latent_dim=latent_dim))

        # for training the encoder
        self.encoder_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")

        self._alpha_tol = 1e-5
        self.forgetting_drift_threshold = forgetting_drift_threshold
        self.forgetting_patience = forgetting_patience
        self.exploration_epochs = exploration_epochs
        self.cooldown_periods = (
            cooldown_periods
            if cooldown_periods is not None
            else int(np.ceil(self.n_batches_per_epoch / self.n_batches_for_forgetting_momentum))
        )
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
        self._forgetting_patience_counters: dict[int, int] = {k: 0 for k in self.k_values}
        self._k_in_final_epoch: dict[int, bool] = {k: False for k in self.k_values}
        self._k_final_epoch_start: dict[int, int | None] = {k: None for k in self.k_values}

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
        encoder_loadings_rnk = getattr(self, f"encoder_{k}")(x_ng, factors_rkg.detach())

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
            n_iter = self.max_solver_iter_cooldown if self._k_in_final_epoch[k] else self.max_solver_iter_train
            out = self.online_dictionary_update(x_ng=x_ng, k=k, n_iterations=n_iter)
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
                # squared_error_r = compute_reconstruction_error_compiled(
                #     x_ng=x_ng,
                #     loadings_rnk=solver_loadings_rnk,
                #     factors_rkg=factors_rkg,
                # )
                squared_error_r = frobenius_loss_trace_compiled(
                    x_ng=x_ng,
                    h_rnk=solver_loadings_rnk,
                    w_rkg=factors_rkg,
                )
                nmf_reconstruction_error = squared_error_r.mean() / (x_ng.shape[0] * x_ng.shape[1])
                nmf_reconstruction_errors.append(nmf_reconstruction_error)

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
        if step == 0 or step % self.n_batches_for_forgetting_momentum != 0:
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

        # --- per-k forgetting convergence check and forgetting ---
        assert isinstance(trainer.max_epochs, int)
        steps_remaining = trainer.max_epochs * self.n_batches_per_epoch - step
        min_steps_before_stop = self.exploration_epochs * self.n_batches_per_epoch

        all_done = True
        for k in self.k_values:
            # check if a previously started final epoch has completed
            cooldown_steps = self.cooldown_periods * self.n_batches_for_forgetting_momentum
            if self._k_in_final_epoch[k]:
                k_start = self._k_final_epoch_start[k]
                assert k_start is not None
                if step - k_start >= cooldown_steps:
                    continue  # this k is done; don't touch A/B
                else:
                    all_done = False
                    continue  # still in cooldown; don't forget
            all_done = False

            # force into cooldown if not enough steps remain
            if steps_remaining < cooldown_steps:
                self._k_in_final_epoch[k] = True
                self._k_final_epoch_start[k] = step
                continue  # don't forget; let A/B accumulate

            # compute drift relative to previous period endpoint
            D_rkg = getattr(self, f"D_{k}_rkg").detach()
            prev = self._D_prev_snapshots[k]
            if prev is not None and step >= min_steps_before_stop:
                prev_norm = prev.norm()
                if prev_norm < 1e-8:
                    drift = float("inf")
                else:
                    drift = ((D_rkg - prev).norm() / prev_norm).item()
                module.log(f"k={k}__forgetting_drift", drift, prog_bar=False)

                if drift < self.forgetting_drift_threshold:
                    self._forgetting_patience_counters[k] += 1
                else:
                    self._forgetting_patience_counters[k] = 0

                if self._forgetting_patience_counters[k] >= self.forgetting_patience:
                    self._k_in_final_epoch[k] = True
                    self._k_final_epoch_start[k] = step
                    continue  # don't forget; let A/B accumulate

            # snapshot current D before forgetting
            self._D_prev_snapshots[k] = D_rkg.clone()

            # forget
            getattr(self, f"A_{k}_rkk").zero_()
            getattr(self, f"B_{k}_rkg").zero_()

        n_k_still_training = sum(
            not (
                self._k_in_final_epoch[k]
                and isinstance(self._k_final_epoch_start[k], int)
                and step - self._k_final_epoch_start[k] >= self.n_batches_per_epoch  # type: ignore[operator]
            )
            for k in self.k_values
        )
        module.log("k_training", float(n_k_still_training), prog_bar=True)

        if all_done:
            trainer.should_stop = True
            print("Stopping early: all k values have completed their final epoch")

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
            encoder_loadings_rnk = getattr(self, f"encoder_{k}")(x_ng, factors_rkg.detach())
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
