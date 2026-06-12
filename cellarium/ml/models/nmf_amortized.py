from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy

from cellarium.ml.models.nmf import (
    NonNegativeMatrixFactorization, 
    NMFInit, 
    NMFInitSklearnRandom, 
    NMFInitUniformRandom,
    compute_reconstruction_error_compiled,
    online_dictionary_update_nmf_torch_hals,
    solve_nnls_fista,
    nmf_frobenius_loss,
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


class AmortizedOnlineNonNegativeMatrixFactorization(NonNegativeMatrixFactorization):
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
        init: Literal["sklearn_random", "uniform_random"] = "uniform_random",
        transformed_data_mean: None | float = None,
    ) -> None:
        super().__init__(var_names_g=var_names_g, k_values=k_values)
        g = len(self.var_names_g)
        self.obs_names_to_index_map: dict[str, int] = {}  # used for local latents
        self.r = r
        self.transformed_data_mean = transformed_data_mean
        self.init = init
        if init == "sklearn_random":
            if transformed_data_mean is None:
                raise ValueError("transformed_data_mean must be provided when using the sklearn_random initialization")

        for i in self.k_values:
            self.register_buffer(f"A_{i}_rkk", torch.empty(r, i, i))
            self.register_buffer(f"B_{i}_rkg", torch.empty(r, i, g))
            self.register_buffer(f"D_{i}_rkg", torch.empty(r, i, g))

            # handle the encoders
            assert len(encoder_hidden_dims) > 0, "encoder_hidden_dims must be a non-empty list of hidden layer dimensions"
            k_encoder = ConsensusNMFEncoder(
                num_genes=g,
                hidden_dims=encoder_hidden_dims,
                num_factors=i,
                num_replicates=r,
            )
            self.add_module(f"encoder_{i}", k_encoder)

        # for training the encoder
        self.encoder_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")

        self._D_tol = 1e-5
        self._alpha_tol = 1e-5
        self._hals_tol = 1e-4
        # Sentinel parameter used to track the current device (mirrors OnlineNMF pattern)
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
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

        self._prev_err_rk: torch.Tensor | None = None
        self._init_err_rk: torch.Tensor | None = None
        self._err_running_sum_rk = torch.zeros((self.r, len(self.k_values)), device=self._dummy_param.device)
        self._cells_seen_in_epoch = 0  # Track cells seen in current epoch

    @property
    def factors_dict(self) -> dict[int, torch.Tensor]:
        """Return the learned factors for each k value."""
        return {k: getattr(self, f"D_{k}_rkg") for k in self.k_values}

    def online_dictionary_update(self, x_ng: torch.Tensor, k: int) -> torch.Tensor:
        """
        Algorithm 1 from Mairal et al. [1] for online dictionary learning.

        Args:
            x_ng: The data.
            k: The value of k to run.
            minibatch_indices_n: The indices of the cells in the current minibatch.

        Returns:
            Loss for the encoder based on HALS targets.
        """
        # get running values
        A_rkk = getattr(self, f"A_{k}_rkk")
        B_rkg = getattr(self, f"B_{k}_rkg")
        factors_rkg = getattr(self, f"D_{k}_rkg")

        # get seed loading values from encoder (rather than from memory)
        encoder_loadings_rnk = getattr(self, f"encoder_{k}")(x_ng)
        hals_loadings_rnk = encoder_loadings_rnk.clone()

        # run nmf-torch hals online update
        updated_values = online_dictionary_update_nmf_torch_hals(
            x_ng=x_ng,
            factors_rkg=factors_rkg,
            loadings_rnk=hals_loadings_rnk,
            A_rkk=A_rkk,
            B_rkg=B_rkg,
            # n_iterations=100,
            # alpha_tol=self._alpha_tol,
            # D_tol=self._D_tol,
        )

        # update running values
        setattr(self, f"A_{k}_rkk", updated_values["A_rkk"])
        setattr(self, f"B_{k}_rkg", updated_values["B_rkg"])
        setattr(self, f"D_{k}_rkg", updated_values["factors_rkg"])

        # hals_loadings_rnk gets updated in-place by online_dictionary_update_nmf_torch_hals
        # this is now the encoder's target
        target_loadings_rnk = F.normalize(hals_loadings_rnk, p=1, dim=-1, eps=1e-8)
        return self.encoder_loss_fn(encoder_loadings_rnk, target_loadings_rnk)

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

        # for error computation to assess convergence
        if self._init_err_rk is None:
            self._loss(x_ng=x_ng)
            # Take sqrt and normalize by number of cells in this batch
            self._init_err_rk = self._err_running_sum_rk.clone() / x_ng.shape[0]
            assert isinstance(self._init_err_rk, torch.Tensor)
            self._prev_err_rk = self._init_err_rk.clone()
            self._err_running_sum_rk.zero_()  # Reset after initialization
            self._cells_seen_in_epoch = 0  # Reset counter

        losses = []
        for k in self.k_values:
            losses.append(self.online_dictionary_update(x_ng=x_ng, k=k))

        # for error computation to assess convergence
        self._loss(x_ng=x_ng)

        return {"loss": sum(losses) / len(losses) if losses else None}

    # @torch.compile()
    @torch.no_grad()
    def _loss(self, x_ng: torch.Tensor) -> None:
        """
        Simple and efficient NMF reconstruction loss computation.
        Computes ||X - WH||_F^2 for the current batch.
        """
        for i, k in enumerate(self.k_values):
            factors_rkg = getattr(self, f"D_{k}_rkg")  # (r, k, g)
            loadings_rnk = getattr(self, f"encoder_{k}")(x_ng)  # (r, n, k)

            squared_error_r = compute_reconstruction_error_compiled(
                x_ng=x_ng, 
                loadings_rnk=loadings_rnk, 
                factors_rkg=factors_rkg,
            )  # (r,)

            # Accumulate the squared error
            self._err_running_sum_rk[:, i] += squared_error_r

        # Track cells seen in this epoch
        self._cells_seen_in_epoch += x_ng.shape[0]

    def on_train_start(self, trainer: pl.Trainer) -> None:
        if trainer.world_size > 1:
            assert isinstance(trainer.strategy, DDPStrategy), (
                "OnlineNonNegativeMatrixFactorization requires that the trainer uses the DDP strategy."
            )
            assert trainer.strategy._ddp_kwargs["broadcast_buffers"] is True, (
                "OnlineNonNegativeMatrixFactorization requires that the `broadcast_buffers` parameter of "
                "lightning.pytorch.strategies.DDPStrategy is set to True"
            )

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        # Take sqrt of accumulated squared errors and normalize by number of cells seen
        # cur_err_rk = torch.sqrt(self._err_running_sum_rk / self._cells_seen_in_epoch)
        cur_err_rk = self._err_running_sum_rk / self._cells_seen_in_epoch
        assert isinstance(self._prev_err_rk, torch.Tensor)
        assert isinstance(self._init_err_rk, torch.Tensor)

        current_overall_err_rk = torch.abs((self._prev_err_rk - cur_err_rk) / self._init_err_rk)
        if current_overall_err_rk.max() < self._hals_tol:
            trainer.should_stop = True
            print(f"Stopping early: converged, loss={cur_err_rk}")

        # print(f"Epoch {trainer.current_epoch} convergence stat: {current_overall_err_rk.max()}")
        # print(f"Per-cell loss - Current max: {cur_err_rk.max():.6f}, Previous: {self._prev_err_rk.max():.6f}")
        # print(f"Per-cell loss - Current mean: {cur_err_rk.mean():.6f}, Previous: {self._prev_err_rk.mean():.6f}")
        self._prev_err_rk = cur_err_rk.clone()
        self._err_running_sum_rk.zero_()
        self._cells_seen_in_epoch = 0  # Reset for next epoch

        # this hard reset to zero is equivalent to forgetting momentum each epoch
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
