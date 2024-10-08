# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import lightning.pytorch as pl
import numpy as np
import pyro
import pyro.distributions
import torch
import torch.nn.functional
import torchmetrics
from pyro.distributions import Delta, Laplace

from cellarium.ml.data.fileio import read_pickle_from_gcs
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
        loss_fn: str|None,
        n_obs: int,
        var_names_g: np.ndarray,
        y_categories: np.ndarray|None,
        alpha: float = 0.5 | None,
        W_prior_scale: float = 1.0,
        W_init_scale: float = 1.0,
        activation_fn: str = 'softmax',
        out_distribution: str = 'Categorical',
        seed: int = 0,
        probability_propagation_flag: bool = False,
        normalize_post_propagation: bool = False,
        parent_child_list_path: str = 'gs://cellarium-file-system/ml-configs/Supervised_cell_classification/tdigest_config/parent_child_indices_list.pkl', # Ask Yerdos, where these files can be stored on gcs
        y_categories_path: str = 'gs://cellarium-file-system/ml-configs/Supervised_cell_classification/tdigest_config/unique_cell_types.pkl',
        log_metrics: bool = True,
    ) -> None:
        super().__init__()

        # data
        self.alpha=alpha
        self.n_obs = n_obs
        self.var_names_g = var_names_g
        self.n_vars = len(var_names_g)
        #self.y_categories = y_categories
        self.y_categories = read_pickle_from_gcs(y_categories_path)
        self.parent_child_list = read_pickle_from_gcs(parent_child_list_path)
        self.n_categories = len(self.y_categories)
        self.activation_fn = getattr(torch.nn.functional, activation_fn)
        self.probability_propagation_flag = probability_propagation_flag
        self.normalize_post_propagation = normalize_post_propagation
        self.loss_fn = loss_fn
        self.out_distribution = getattr(pyro.distributions,out_distribution)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=len(self.y_categories))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self, x_ng: torch.Tensor, var_names_g: np.ndarray, y_n: torch.Tensor
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
        print(f"NIMISH SELF.ACTIVATION_FN is {self.activation_fn}")
        print(f"NIMISH SELF.OUT_DISTRIBUTION is {self.out_distribution}")
        print(f"NIMISH SELF.PP FLAG is {self.probability_propagation_flag}")
        y_n = y_n.to(self.device)
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "self.var_names_g", self.var_names_g)
        #assert_arrays_equal("y_categories", y_categories, "self.y_categories", self.y_categories)
        loss = self.elbo.differentiable_loss(self.model, self.guide, x_ng, y_n)
        return {"loss": loss}

    def model(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        W_gc = pyro.sample(
            "W",
            Laplace(0, self.W_prior_scale).expand([self.n_vars, self.n_categories]).to_event(2),
        )
        with pyro.plate("batch", size=self.n_obs, subsample_size=x_ng.shape[0]):
            logits_nc = x_ng @ W_gc + self.b_c
            activation_out = self.activation_fn(logits_nc.to(dtype=torch.float64), dim=1)
            if (self.probability_propagation_flag==1):
                #activation_out = custom_functions.multi_label_target(pp_flag=1,softmax_out_gpu=activation_out)
                activation_out = self.probability_propagation(activation_out_gpu=activation_out)

            print(f"NIMISH SHAPE OF ACTIVATION OUT IS {activation_out.shape}")
            print(f"NIMISH SHAPE OF Y_N IS {y_n.shape}")
            print(f"TYPE OF Y_N IS {type(y_n)}")
            if self.out_distribution == pyro.distributions.Categorical:
                pyro.sample("y", self.out_distribution(probs=torch.round(activation_out,decimals=5)), obs=torch.argmax(y_n, dim=-1))
            elif self.out_distribution == pyro.distributions.Bernoulli:
                pyro.sample("y", self.out_distribution(probs=torch.round(activation_out,decimals=5)).to_event(1), obs=y_n.float())

    def guide(self, x_ng: torch.Tensor, y_n: torch.Tensor) -> None:
        pyro.sample("W", Delta(self.W_gc).to_event(2))

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
        return {"y_logits_nc": logits_nc}

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
        activation_out_gpu_clone = activation_out_gpu.clone()
        for col in range(activation_out_gpu.shape[1]):
            activation_out_gpu_clone[:, col] += activation_out_gpu[torch.arange(activation_out_gpu_clone.shape[0]).unsqueeze(1),self.parent_child_list[col]].sum(dim=1)
        return activation_out_gpu_clone



    # def probability_propagation(self,softmax_out_gpu:torch.tensor):
    #     softmax_out_gpu_clone = softmax_out_gpu.clone() #cannot modify softmax output tensor in place due to gradients
    #     for i in range(softmax_out_gpu.shape[1]):
    #         softmax_out_gpu_clone[:][i].add_(softmax_out_gpu[torch.arange(softmax_out_gpu.shape[0].unsqueeze(1)),self.parent_child_list[i]].sum(dim=1))
    #     for i in range(softmax_out_gpu.shape[0]):
    #         for j in range(softmax_out_gpu.shape[1]):
    #             #children_indices = np.where(child_parent_array.transpose()[j] == 1)[0] #list of child cell type indices
    #             softmax_out_gpu_clone[i][j].add_(softmax_out_gpu[i][self.parent_child_list[j]].sum()) #parent cell type probability is sum of all child cell type probabilities
    #         #softmax_out_gpu_clone[i][:] = softmax_out_gpu_clone[i][:]/torch.sum(softmax_out_gpu[i][:]) #normalization step after probability propagation
    #         if (softmax_out_gpu_clone[i][j]>1).any():
    #             print(f"FOUND ERROR VALUE AT INDICES {i,j} AND VALUE IS {softmax_out_gpu_clone[i][j]}")
    #     print("RETURNING SOFTMAX OUT GPU CLONE")
    #     return softmax_out_gpu_clone

    def validate(self,x_ng: torch.Tensor,y_n: torch.Tensor,pl_module: pl.LightningModule) -> None:
        logits_nc = x_ng @ self.W_gc + self.b_c
        y_hat = torch.argmax(logits_nc, dim=-1)
        f1_score = self.f1(y_hat, y_n)
        pl_module.log("F1_score_val",f1_score,sync_dist=True, on_epoch=True)
        return None
