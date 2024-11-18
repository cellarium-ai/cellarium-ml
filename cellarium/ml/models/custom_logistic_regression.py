# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import lightning.pytorch as pl
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional

from cellarium.ml.categorical_distribution import bernoulli_distribution, categorical_distribution
from cellarium.ml.data.fileio import read_pkl_from_gcs
from cellarium.ml.models.model import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.utilities.testing import (
    assert_arrays_equal,
    assert_columns_and_array_lengths_equal,
)


class CustomLogisticRegression(CellariumModel, PredictMixin, ValidateMixin):
    """
    Logistic regression model.

    Args:
        n_obs:
            Number of observations.
        var_names_g:
            The variable names schema for the input data validation.
        y_categories:
            The categories for the target data.
        W_prior_scale:
            The scale of the Laplace prior for the weights.
        W_init_scale:
            Initialization scale for the ``W_gc`` parameter.
        seed:
            Random seed used to initialize parameters.
        log_metrics:
            Whether to log the histogram of the ``W_gc`` parameter.
    """

    def __init__(
        self,
        n_obs: int,
        var_names_g: np.ndarray,
        y_categories: np.ndarray | None,
        W_prior_scale: float = 1.0,
        W_init_scale: float = 1.0,
        activation_fn: str = 'softmax',
        out_distribution: str = 'Categorical',
        seed: int = 0,
        probability_propagation_flag: bool = False,
        target_row_descendent_col_torch_tensor_path: str = 'gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/target_row_descendent_col_torch_tensor.pkl',
        #target_row_descendent_col_torch_tensor_path: str = 'gs://cellarium-file-system/curriculum/lrexp_human_training_split_20241106/models/shared_metadata/target_row_descendent_col_torch_tensor_lrexp_human.pkl',
        y_categories_path: str = 'gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_filtered_sorted_unique_cells.pkl',
        #y_categories_path: str = 'gs://cellarium-file-system/curriculum/lrexp_human_training_split_20241106/models/shared_metadata/final_filtered_sorted_unique_cells_lrexp_human.pkl',
        log_metrics: bool = True,
    ) -> None:
        super().__init__()

        # data
        self.n_obs = n_obs
        self.var_names_g = var_names_g
        self.n_vars = len(var_names_g)
        #self.y_categories = y_categories
        self.y_categories = read_pkl_from_gcs(y_categories_path)
        self.target_row_descendent_col_torch_tensor = read_pkl_from_gcs(target_row_descendent_col_torch_tensor_path)
        self.n_categories = len(self.y_categories)
        self.activation_fn = getattr(torch.nn.functional, activation_fn)
        self.probability_propagation_flag = probability_propagation_flag
        if out_distribution == "Categorical":
            self.out_distribution = getattr(categorical_distribution,'Pyro'+out_distribution)
        else:
            #self.out_distribution = getattr(dist,out_distribution)
            self.out_distribution = getattr(bernoulli_distribution,'CustomPyro'+out_distribution)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = seed
        # parameters
        self._W_prior_scale = W_prior_scale
        self.W_init_scale = W_init_scale
        self.W_prior_scale: torch.Tensor
        self.register_buffer("W_prior_scale", torch.empty(()))
        self.W_gc = torch.nn.Parameter(torch.empty(self.n_vars, self.n_categories))
        self.b_c = torch.nn.Parameter(torch.empty(self.n_categories))
        self.reset_parameters()

        # loss
        self.elbo = pyro.infer.Trace_ELBO()

        self.log_metrics = log_metrics

    def reset_parameters(self) -> None:
        rng_device = self.W_gc.device.type if self.W_gc.device.type != "meta" else "cpu"
        rng = torch.Generator(device=rng_device)
        rng.manual_seed(self.seed)
        self.W_prior_scale.fill_(self._W_prior_scale)
        self.W_gc.data.normal_(0, self.W_init_scale, generator=rng)
        self.b_c.data.zero_()

    def forward(
        self, x_ng: torch.Tensor, var_names_g: np.ndarray, y_n: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x_ng:
                The input data.
            var_names_g:
                The variable names for the input data.
            y_n:
                The target data.
            y_categories:
                The categories for the input target data.

        Returns:
            A dictionary with the loss value.
        """
        #y_n = y_n.to(self.device)
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        #assert_arrays_equal("y_categories", y_categories, "self.y_categories", self.y_categories)
        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n)
        return {"loss": loss}

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        W_gc = pyro.sample(
            "W",
            dist.Laplace(0, self.W_prior_scale).expand([self.n_vars, self.n_categories]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0]):
            logits_nc = x_ng @ W_gc + self.b_c
            compiled_propagated_logits = torch.compile(self.log_probs)
            propagated_logits = compiled_propagated_logits(logits=logits_nc)
            #activation_out = self.activation_fn(logits_nc.to(dtype=torch.float), dim=1)
            #if (self.probability_propagation_flag==1):
                #activation_out = self.probability_propagation(activation_out_gpu=activation_out)
            if self.out_distribution == categorical_distribution.PyroCategorical:
                #compiled_propagated_logits = torch.compile(self.log_probs)
                #propagated_logits = compiled_propagated_logits(logits=logits_nc)
                pyro.sample("y", self.out_distribution(logits = propagated_logits), obs=y_n)
                #pyro.sample("y", dist.Categorical(probs=activation_out), obs=y_n)
            elif self.out_distribution == dist.Bernoulli:
                logits_complement = self.bernoulli_log_probs(logits_nc=logits_nc, dim=0, keepdim=False)
                #activation_out = self.activation_fn(logits_nc.to(dtype=torch.float), dim=1)
                #activation_out = self.probability_propagation(activation_out_gpu=activation_out)
                #pyro.sample("y", self.out_distribution(probs=activation_out).to_event(1), obs=y_n)
                #print(f"NIMISH PROPAGATED LOGITS ARE {propagated_logits}")
                pyro.sample("y", self.out_distribution(logits = propagated_logits,logits_complement=logits_complement).to_event(1), obs=y_n)

    def guide(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        pyro.sample("W", dist.Delta(self.W_gc).to_event(2))

    def predict(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Predict the target logits.

        Args:
            x_ng:
                The input data.
            var_names_g:
                The variable names for the input data.

        Returns:
            A dictionary with the target logits.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)

        logits_nc = x_ng @ self.W_gc + self.b_c
        #Option 1
        activation_out = torch.nn.functional.softmax(logits_nc.to(dtype=torch.float), dim=1)
        activation_out = self.probability_propagation(activation_out_gpu=activation_out)
        #Option 2
        # return {"y_logits_nc": logits_nc,"cell_type_probs_nc": activation_out}
        compiled_propagated_logits = torch.compile(self.log_probs)
        propagated_logits = compiled_propagated_logits(logits=logits_nc)

        #applying softmax to propagated logits
        return {"y_logits_nc": propagated_logits,"cell_type_probs_nc": activation_out}

    def on_train_batch_end(self, trainer: pl.Trainer) -> None:
        if trainer.global_rank != 0:
            return

        if not self.log_metrics:
            return

        if (trainer.global_step + 1) % trainer.log_every_n_steps != 0:  # type: ignore[attr-defined]
            return

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_histogram(
                    "W_gc",
                    self.W_gc,
                    global_step=trainer.global_step,
                )

    def probability_propagation(self,activation_out_gpu:torch.tensor) -> torch.tensor:
        """
        for each column in activation_out_gpu_clone, col, we get a tensor of
        activation_out_gpu[:,[children_indices[col]]],
        then we take the rowwise sume of these columns.
        then we add those values with all rows in 'col' column in activation_out_gpu_clone
        TRIAL CODE AVAILABLE IN TRIAL.IPYNB - PROBABILITY PROPAGATION CODE TRIAL
        """
        propagated_p = torch.einsum("nc,kc->nk", activation_out_gpu, self.target_row_descendent_col_torch_tensor.to(device=activation_out_gpu.device))
        return torch.clamp(propagated_p, max=1.0)

    def log_probs(self, logits: torch.Tensor):
        """
        logits = torch Tensor of shape nxc
        step 1: propagated_logits: LSE (L_i) where i belongs to the set of descendents of each column in c and column c
        step 2: logits_rowwise_sum: LSE (L_m) where m belongs to the 2604 classes c in each row (sum of all logits in each row of n before propagation)
        log(1-p_k') = log{(sum(p_i) i belongs to set(descendents(k)) + p_k)/sum(p_m), m = total cell types}
        step 3: LSE(L_i) - LSE(L_m)
        """
        log_probs = self.logsumexp_propagated(logits,dim=0,keepdim=False) - torch.logsumexp(logits,dim=1,keepdim=True)
        return log_probs

    #  OPTION 1
    # def logsumexp_propagated(self, logits: torch.Tensor, dim, keepdim=False):
    #     desc_matrix = self.target_row_descendent_col_torch_tensor.to(device=logits.device, dtype = torch.float)
    #     desc_matrix_1 = torch.where(desc_matrix == 0, float('-inf'), desc_matrix)

    #     #max_values = torch.empty((logits.shape[0], logits.shape[1])).to(device=logits.device)
    #     log_sum_exp_nc = torch.empty((logits.shape[0], logits.shape[1])).to(device=logits.device)
    #     chunk_size=2
    #     # Loop through each row in `b` to get the corresponding masked maximum

    #     for i in range(0,logits.shape[0],chunk_size):
    #         with torch.no_grad():
    #             max_values_chc = torch.amax(torch.nan_to_num(masked_values_chcc,float('-inf')), dim=2)
    #         exp_values = torch.exp(logits[i:i+chunk_size,:] - max_values_chc)
    #         sum_exp = torch.einsum("nc,kc->nk", exp_values, desc_matrix)
    #         log_sum_exp = torch.log(sum_exp) + max_values_chc if not keepdim else max_values_chc
    #         log_sum_exp_nc[i:i+chunk_size,:] = log_sum_exp
    #     return log_sum_exp_nc

    # OPTION 2
    # def logsumexp_propagated(self, logits_nc: torch.Tensor, dim, keepdim=False):
    #     max_values_nc = torch.empty((logits_nc.shape[0], logits_nc.shape[1])).to(device=logits_nc.device)
    #     desc_matrix_cc = self.target_row_descendent_col_torch_tensor.to(device=logits_nc.device, dtype = torch.float)
    #     with torch.no_grad():
    #         for i,row in enumerate(desc_matrix_cc):
    #             masked_logits_nc = torch.where(row.to(dtype=torch.bool), logits_nc, float('-inf'))
    #             max_values_nc[:,i] = torch.max(masked_logits_nc, dim=1).values.T
    #     exp_values_nc = torch.exp(logits_nc - max_values_nc)

    #     # Sum the exponentials using einsum
    #     sum_exp_nc = torch.einsum("nc,kc->nk", exp_values_nc, desc_matrix_cc)
    #     #sum_exp_sparse = torch.sparse.mm( exp_values_nc, desc_matrix_cc_sparse.T)

    #     log_sum_exp_nc = torch.log(sum_exp_nc) + max_values_nc.squeeze(dim) if not keepdim else max_values_nc
    #     return log_sum_exp_nc

    # OPTION 3
    def logsumexp_propagated(self, logits_nc: torch.Tensor, dim, keepdim=False):
        desc_matrix_cc = self.target_row_descendent_col_torch_tensor.to(device=logits_nc.device, dtype = torch.float)
        desc_matrix_1 = torch.where(desc_matrix_cc == 0, float('-inf'), desc_matrix_cc)
        with torch.no_grad():
            max_values_nc = torch.amax(torch.nan_to_num(logits_nc.unsqueeze(1) * desc_matrix_1.unsqueeze(0), nan=float('-inf'), posinf=float('-inf'), neginf=float('-inf')),dim=2)
        exp_values_nc = torch.exp(logits_nc - max_values_nc)
        sum_exp_nc = torch.einsum("nc,kc->nk", exp_values_nc, desc_matrix_cc)
        log_sum_exp_nc = torch.log(sum_exp_nc) + max_values_nc.squeeze(dim) if not keepdim else max_values_nc
        return log_sum_exp_nc

    def bernoulli_log_probs(self, logits_nc: torch.Tensor, dim, keepdim=False):
        desc_matrix_cc = self.target_row_descendent_col_torch_tensor.to(device=logits_nc.device, dtype = torch.float)
        max_values_nc = torch.amax(logits_nc,dim=1)
        exp_values_nc = torch.exp(logits_nc - max_values_nc)
        #sum_exp_nc_1 = torch.sum(exp_values_nc, dim=1)
        sum_exp_nc_2 = torch.einsum("nc,kc->nk", exp_values_nc, desc_matrix_cc)
        log_sum_exp_nc = torch.log(exp_values_nc - sum_exp_nc_2) + max_values_nc.squeeze(dim) if not keepdim else max_values_nc
        logits_prob_b = log_sum_exp_nc - torch.logsumexp(logits_nc,dim=1,keepdim=True)
        return logits_prob_b



