# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Imputation model for missing genes in single-cell RNA-seq data."""

import logging

import numpy as np
import torch
import math
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from cellarium.ml.models.scvi import SingleCellVariationalInference, compute_annealed_kl_weight
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)

logger = logging.getLogger(__name__)


class ImputationModel(SingleCellVariationalInference):
    """Imputation model for missing genes in single-cell RNA-seq data.

    This model extends the SingleCellVariationalInference class to provide
    functionality for imputing missing gene expression values in single-cell
    RNA-seq datasets.
    """

    def __init__(self, *args, **kwargs):

        self.masking_probability = kwargs.pop("masking_probability", 0.5)

        # set boolean flag for whether to anneal noise2self ratio during training
        self.anneal_noise2self = kwargs.pop("anneal_noise2self", True)
        
        # anneal_noise2self is false set weight to max value right away
        self.noise2self_weight = kwargs.pop("noise2self_weight", 0.5) if not self.anneal_noise2self else None

        # noise2self annealing parameters  
        self.noise2self_ratio_max = kwargs.pop("noise2self_ratio_max", 0.9)
        self.noise2self_annealing_start = kwargs.pop("noise2self_annealing_start", 0.0)
        self.noise2self_warmup_epochs = kwargs.pop("noise2self_warmup_epochs", 400)
        self.noise2self_warmup_steps = kwargs.pop("noise2self_warmup_steps", None)

        if not (self.noise2self_annealing_start >= 0.0 and self.noise2self_annealing_start <= 1.0):
            raise ValueError(f"noise2self_annealing_start={self.noise2self_annealing_start} must be in the range [0.0, 1.0].")
        assert not ((self.noise2self_warmup_steps is not None) and (self.noise2self_warmup_epochs is not None)), (
            "Only one of noise2self_warmup_epochs or noise2self_warmup_steps can be specified, not both."
        )

        # Gene list to exclude from masking
        print("Genes to exclude from masking (if any):", kwargs.get("gene_to_exclude_from_masking", None))
        self.gene_to_exclude_from_masking = kwargs.pop("gene_to_exclude_from_masking", None)
        super().__init__(*args, **kwargs)
        self._excluded_gene_indices = self.exclude_genes_from_masking()


    def exclude_genes_from_masking(self) -> list[int]:
        """Update the gene mask to exclude specified genes from being masked during imputation.

        Returns:
            List of indices of genes to exclude from masking
        """
        excluded_gene_indices: list[int] = []
        if self.gene_to_exclude_from_masking is not None:
            for gene in self.gene_to_exclude_from_masking:
                matches = np.where(self.var_names_g == gene)[0]
                if len(matches) == 0:
                    continue
                excluded_gene_indices.append(int(matches[0]))
        return excluded_gene_indices

    def create_gene_mask(self, n_genes: int, device: torch.device) -> torch.Tensor:
        """Create a random gene mask for imputation.
        
        Args:
            n_genes: Number of genes in the dataset
            device: Device to create the mask on
            
        Returns:
            Boolean tensor of shape (n_genes,) where True indicates genes to be masked
        """

        # Start with all genes available
        available_mask = torch.ones(n_genes, device=device, dtype=torch.bool)

        # Remove excluded genes from availability
        if self._excluded_gene_indices:
            available_mask[self._excluded_gene_indices] = False
        
        # Get indices of genes that are allowed to be masked
        available_indices = torch.where(available_mask)[0]
        n_available = len(available_indices)

        # Randomly sample from allowed genes only
        perm = torch.randperm(n_available, device=device)
        selected = available_indices[perm[: int(self.masking_probability * n_available)]].long()

        # Create final logical mask
        logical_mask_g = torch.zeros(n_genes, device=device, dtype=torch.bool)
        logical_mask_g[selected] = True

        return logical_mask_g

    def apply_gene_mask(self, x_ng: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply gene mask to expression data.
        
        Args:
            x_ng: Gene expression data of shape (n_cells, n_genes)
            mask: Boolean mask of shape (n_genes,) where True indicates genes to mask
            
        Returns:
            Masked gene expression data with masked genes set to 0
        """
        x_masked_ng = x_ng.clone()
        x_masked_ng[:, mask] = 0.0
        return x_masked_ng

    def compute_annealed_noise2self_weight(self, 
                                           epoch: int, 
                                           step: int, 
                                           n_epochs_n2s_warmup: int | None,
                                           n_steps_n2s_warmup: int | None,
                                           max_n2s_weight: float = 1.0,
                                           min_n2s_weight: float = 0.0,
                                           schedule_type: str = "cosine", # "cosine" or "delayed_linear"
                                           delay_fraction: float = 0.3, # used only for delayed_linear
                                        ) -> float:
        """Computes the Noise2Self weight using cosine ramp-up.
        
        If both `n_epochs_n2s_warmup` and `n_steps_n2s_warmup` are None,
        `max_weight` is returned.
        
        Args:
            epoch: Current epoch.
            step: Current step.
            n_epochs_n2s_warmup: Number of epochs to ramp up the weight.
            n_steps_n2s_warmup: Number of steps to ramp up the weight.
            max_weight: Maximum Noise2Self weight.
            min_weight: Minimum Noise2Self weight.
            schedule_type: Type of schedule to use ("cosine" or "delayed_linear").
            delay_fraction: Fraction of the warmup period to delay the start of the ramp-up (used only for delayed_linear).
        
        Returns:
            The Noise2Self weight for the current step or epoch.
        """
        if min_n2s_weight > max_n2s_weight:
            raise ValueError(
                f"min_n2s_weight={min_n2s_weight} is larger than max_n2s_weight={max_n2s_weight}."
            )
        amplitude = max_n2s_weight - min_n2s_weight

        # Determine progress variable
        if n_epochs_n2s_warmup:
            current = epoch
            total = n_epochs_n2s_warmup
        elif n_steps_n2s_warmup:
            current = step
            total = n_steps_n2s_warmup
        else:
            return max_n2s_weight
        # If warmup finished
        if current >= total:
            return max_n2s_weight
        progress = current / total  # normalized 0 → 1

        # Cosine Ramp-up
        if schedule_type == "cosine":
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            return min_n2s_weight + amplitude * cosine_factor
        # Delayed Linear Ramp-up
        elif schedule_type == "delayed_linear":
            if not (0.0 <= delay_fraction < 1.0):
                raise ValueError("delay_fraction must be in [0, 1).")
            if progress < delay_fraction:
                return min_n2s_weight
            # rescale remaining progress to [0,1]
            adjusted_progress = (progress - delay_fraction) / (1 - delay_fraction)
            return min_n2s_weight + amplitude * adjusted_progress
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
        total_mrna_umis_n: torch.Tensor | None = None,
    ):
        """
        Put data through the VAE, but use a noise2self loss, where we compute a randomized mask for each minibatch
        and only compute the reconstruction loss on the masked values.

        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.
            batch_index_n:
                Batch indices of input cells as integers.
            continuous_covariates_nc:
                Continuous covariates for each cell (c-dimensional).
            categorical_covariate_index_nd:
                Categorical covariates for each cell (d-dimensional). Integer membership categorical codes.
            total_mrna_umis_n:
                Total mRNA UMIs for each cell (not used here, but included for compatibility with the parent class).

        Returns:
            A dictionary with keys:
                - "loss": The total loss value.
                - "reconstruction_loss": The reconstruction loss value.
                - "noise2self_rec_loss_n": Reconstruction loss on masked genes (per cell).
                - "noise2self_weight": Annealed weight controlling Noise2Self contribution.
                - "kl_divergence_z": The KL divergence for the latent variable z.
                - "z_nk": The latent variable z.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        # Create mask and apply it to the input data
        logical_mask_g = self.create_gene_mask(x_ng.shape[1], x_ng.device)
        x_masked_ng = self.apply_gene_mask(x_ng, logical_mask_g)

        batch_nb = self.batch_representation_from_batch_index(batch_index_n)
        categorical_covariate_np = self.categorical_onehot_from_categorical_index(categorical_covariate_index_nd)

        if self.use_size_factor_key:
            assert total_mrna_umis_n is not None, "total_mrna_umis_n must be provided when use_size_factor_key=True"
            library_size_n1 = torch.log(total_mrna_umis_n).unsqueeze(-1)
        else:
            library_size_n1 = torch.log(x_ng.sum(dim=-1, keepdim=True))

        inference_outputs = self.inference(
            x_ng=x_masked_ng,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )
        generative_outputs = self.generative(
            z_nk=inference_outputs["z"],
            library_size_n1=library_size_n1,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )

        # KL divergence for z
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=1)

        # optional KL divergence for batch representation
        kl_divergence_batch: torch.Tensor | int
        if self.batch_representation_sampled and (self.batch_kl_weight_max > 0):
            kl_divergence_batch = kl(
                self.batch_embedding_distribution(batch_index_n=batch_index_n),
                Normal(torch.zeros_like(batch_nb), torch.ones_like(batch_nb)),
            ).sum(dim=1)
        else:
            kl_divergence_batch = 0

        # compute the annealed KL weight
        kl_annealing_weight = compute_annealed_kl_weight(
            epoch=self.epoch,
            step=self.step,
            n_epochs_kl_warmup=self.kl_warmup_epochs,
            n_steps_kl_warmup=self.kl_warmup_steps,
            max_kl_weight=1.0,
            min_kl_weight=self.kl_annealing_start,
        )

        # compute the annealed noise2self weight
        self.noise2self_weight = self.compute_annealed_noise2self_weight(
            epoch=self.epoch,
            step=self.step,
            n_epochs_n2s_warmup=self.noise2self_warmup_epochs,
            n_steps_n2s_warmup=self.noise2self_warmup_steps,
            max_n2s_weight=self.noise2self_ratio_max,
            min_n2s_weight=self.noise2self_annealing_start,
        )

        if self.training:
            logger.info(
                f"Epoch {self.epoch}: noise2self_weight = {self.noise2self_weight:.4f}"
            )   

        # compute the regular scvi reconstruction loss
        rec_loss_n = -generative_outputs["px"].log_prob(x_ng)[:, ~logical_mask_g].sum(-1)

        # apply the noise2self mask when computing the reconstruction loss
        noise2self_rec_loss_n = -generative_outputs["px"].log_prob(x_ng)[:, logical_mask_g].sum(-1) 

        # make sure the KL annealing weight is between 0 and 1
        assert kl_annealing_weight >= 0.0 and kl_annealing_weight <= 1.0, (
            f"Invalid KL annealing weight: {kl_annealing_weight}"
        )

        # make sure the noise2self weight is between 0 and 1
        assert self.noise2self_weight >= 0.0 and self.noise2self_weight <= 1.0, (
            f"Invalid Noise2Self weight: {self.noise2self_weight}"
        )   

        # full loss
        loss = torch.mean(
            (1.0 - self.noise2self_weight) * rec_loss_n
            + self.noise2self_weight * noise2self_rec_loss_n
            + kl_annealing_weight
            * (self.z_kl_weight_max * kl_divergence_z + self.batch_kl_weight_max * kl_divergence_batch),
            dim=0,
        )

        return {
            "loss": loss,
            "reconstruction_loss": rec_loss_n,
            "noise2self_rec_loss_n": noise2self_rec_loss_n,
            "noise2self_weight": self.noise2self_weight,
            "kl_divergence_z": kl_divergence_z,
            "kl_annealing_weight": kl_annealing_weight,
            "z_nk": inference_outputs["z"],
        }

    def on_train_epoch_end(self, trainer) -> None:
        noise2self_weight = self.compute_annealed_noise2self_weight(
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            n_epochs_n2s_warmup=self.noise2self_warmup_epochs,
            n_steps_n2s_warmup=self.noise2self_warmup_steps,
            max_n2s_weight=self.noise2self_ratio_max,
            min_n2s_weight=self.noise2self_annealing_start,
        )

        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {"noise2self_weight": noise2self_weight},
                step=trainer.current_epoch,  # epoch-level metric
            )