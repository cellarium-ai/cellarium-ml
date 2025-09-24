"""Imputation model for missing genes in single-cell RNA-seq data."""

from cellarium.ml.models.scvi import SingleCellVariationalInference
import numpy as np
import torch
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal

import logging

from cellarium.ml.models.scvi import compute_annealed_kl_weight

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

        if "masking_probability" in kwargs:
            self.masking_probability = kwargs.pop("masking_probability")
        else:
            self.masking_probability = 0.5  # default masking probability

        super().__init__(*args, **kwargs)

    def create_gene_mask(self, n_genes: int, device: torch.device) -> torch.Tensor:
        """Create a random gene mask for imputation.
        
        Args:
            n_genes: Number of genes in the dataset
            device: Device to create the mask on
            
        Returns:
            Boolean tensor of shape (n_genes,) where True indicates genes to be masked
        """
        # Create random tensor for gene selection
        random_tensor_g = torch.rand(n_genes, device=device)
        # Select indices of genes to mask based on masking probability
        mask_inds_i = torch.argsort(random_tensor_g)[: int(self.masking_probability * n_genes)].long()
        # Create logical mask
        logical_mask_g = torch.zeros(n_genes, device=device).bool()
        logical_mask_g[mask_inds_i] = True
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
        
    def forward(
        self,
        x_ng: torch.Tensor,
        var_names_g: np.ndarray,
        batch_index_n: torch.Tensor,
        continuous_covariates_nc: torch.Tensor | None = None,
        categorical_covariate_index_nd: torch.Tensor | None = None,
        size_factor_n1: torch.Tensor | None = None,
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
            size_factor_n1:
                Library size factor for each cell.

        Returns:
            A dictionary with keys:
                - "loss": The total loss value.
                - "reconstruction_loss": The reconstruction loss value.
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

        inference_outputs = self.inference(
            x_ng=x_masked_ng,
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
        )
        generative_outputs = self.generative(
            z_nk=inference_outputs["z"],
            library_size_n1=inference_outputs["library_size_n1"],
            batch_nb=batch_nb,
            continuous_covariates_nc=continuous_covariates_nc,
            categorical_covariate_np=categorical_covariate_np,
            size_factor_n1=size_factor_n1,
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

        # apply the noise2self mask when computing the reconstruction loss
        rec_loss_n = -generative_outputs["px"].log_prob(x_ng)[
            :, logical_mask_g
        ].sum(-1) / max(1, logical_mask_g.sum())

        # full loss
        assert kl_annealing_weight >= 0.0 and kl_annealing_weight <= 1.0, (
            f"Invalid KL annealing weight: {kl_annealing_weight}"
        )
        loss = torch.mean(
            rec_loss_n
            + kl_annealing_weight
            * (self.z_kl_weight_max * kl_divergence_z + self.batch_kl_weight_max * kl_divergence_batch),
            dim=0,
        )

        return {
            "loss": loss,
            "reconstruction_loss": rec_loss_n,
            "kl_divergence_z": kl_divergence_z,
            "z_nk": inference_outputs["z"],
        }
