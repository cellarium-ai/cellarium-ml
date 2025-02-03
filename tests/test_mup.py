# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import math
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import lightning.pytorch as pl
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from tenacity import retry, stop_after_attempt, wait_fixed
from torch import nn
from torchvision import datasets, transforms

from cellarium.ml.layers import MuLinear
from cellarium.ml.utilities.layers import create_initializer, scale_initializers_by_dimension
from cellarium.ml.utilities.mup import LRAdjustmentGroup
from cellarium.ml.utilities.testing import coord_check_MLP, get_coord_data

optim_dict = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


# adapted from https://github.com/microsoft/mup/blob/main/examples/MLP/main.py
def get_coord_data_MLP(
    implementation: Literal["sp", "mup_mu_linear", "mup_cerebras"],
    bias: bool,
    nonlin: Callable[[torch.Tensor], torch.Tensor],
    lr: float,
    input_mult: float,
    output_mult: float,
    optim_name: Literal["sgd", "adam", "adamw"],
    train_loader: torch.utils.data.DataLoader,
    nsteps: int,
    nseeds: int,
    widths: list[int],
) -> pd.DataFrame:
    """
    Coordinate checking for an MLP model.

    Args:
        implementation:
            Implementation of the model.
        bias:
            If ``True``, use bias in the model.
        nonlin:
            Nonlinearity function to use in the model.
        lr:
            Learning rate.
        input_mult:
            Input multiplier.
        output_mult:
            Output multiplier.
        optim_name:
            Name of the optimizer to use.
        train_loader:
            Training data loader.
        nsteps:
            Number of steps to train for.
        nseeds:
            Number of repeats with different random seeds.
        widths:
            List of widths of the models to use for coordinate checking.

    Returns:
        Dataframe with the results of the coordinate checking.
    """
    optim_fn = optim_dict[optim_name]

    def gen(w: int) -> Callable[[], pl.LightningModule]:
        def f() -> pl.LightningModule:
            model: pl.LightningModule
            if implementation == "sp":
                model = MLP(
                    width=w,
                    bias=bias,
                    nonlin=nonlin,
                    input_mult=input_mult,
                    output_mult=output_mult,
                    loss_fn=F.cross_entropy,
                    optim_fn=optim_fn,
                    lr=lr,
                )
            elif implementation == "mup_mu_linear":
                model = MuLinearMLP(
                    width=w,
                    bias=bias,
                    nonlin=nonlin,
                    optimizer=optim_name,
                    input_mult=input_mult,
                    output_mult=output_mult,
                    loss_fn=F.cross_entropy,
                    optim_fn=optim_fn,
                    lr=lr,
                )
            elif implementation == "mup_cerebras":
                model = CerebrasMLP(
                    width=w,
                    bias=bias,
                    nonlin=nonlin,
                    input_mult=input_mult,
                    output_mult=output_mult,
                    loss_fn=F.cross_entropy,
                    optim_fn=optim_fn,
                    lr=lr,
                )
            return model

        return f

    models = {w: gen(w) for w in widths}
    layer_name_to_multiplier_name = {"fc_1": "input_mult", "fc_3": "output_mult"}

    return get_coord_data(
        models,
        layer_name_to_multiplier_name,
        train_loader,
        nseeds=nseeds,
        nsteps=nsteps,
    )


@retry(
    stop=stop_after_attempt(3),  # retry up to 3 times
    wait=wait_fixed(10),  # wait 10 seconds between retries
    retry=(lambda exc: isinstance(exc, urllib.error.URLError)),  # retry on URLError
)
def cifar_dataset(path: Path, transform) -> datasets.CIFAR10:
    return datasets.CIFAR10(root=path, train=True, download=True, transform=transform)


@pytest.fixture
def train_loader(tmp_path: Path) -> torch.utils.data.DataLoader:
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = cifar_dataset(tmp_path, transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


class MLP(pl.LightningModule):
    """
    A Multi-Layer Perceptron (MLP) model with 3 layers in standard parameterization (SP).

    Args:
        width:
            Width of the hidden layer.
        num_classes:
            Number of classes.
        bias:
            If ``True``, use bias in the model.
        nonlin:
            Nonlinearity function to use in the model.
        input_mult:
            Input multiplier.
        output_mult:
            Output multiplier.
        loss_fn:
            Loss function to use.
        optim_fn:
            Optimizer to use.
        lr:
            Learning rate.
        eps:
            Epsilon value for the optimizer.
        weight_decay:
            Weight decay value for the optimizer.
    """

    def __init__(
        self,
        width: int = 128,
        num_classes: int = 10,
        bias: bool = False,
        nonlin: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
        optim_fn: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.width = width
        self.bias = bias
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

        self.fc_1 = nn.Linear(3072, width, bias=bias)
        self.fc_2 = nn.Linear(width, width, bias=bias)
        self.fc_3 = nn.Linear(width, num_classes, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in_1 = self.fc_1.weight.shape[1]
        nn.init.normal_(self.fc_1.weight, std=1 / math.sqrt(fan_in_1))  # 1 / sqrt(d)
        self.fc_1.weight.data /= self.input_mult
        fan_in_2 = self.fc_2.weight.shape[1]
        nn.init.normal_(self.fc_2.weight, std=1 / math.sqrt(fan_in_2))  # 1 / sqrt(n)
        nn.init.zeros_(self.fc_3.weight)  # zero readout
        if self.bias:
            # zero biases
            nn.init.zeros_(self.fc_1.bias)
            nn.init.zeros_(self.fc_2.bias)
            nn.init.zeros_(self.fc_3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlin(self.fc_1(x) * self.input_mult)
        x = self.nonlin(self.fc_2(x))
        return self.fc_3(x) * self.output_mult

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data.view(data.size(0), -1))
        loss = self.loss_fn(output, target)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim_kwargs: dict[str, Any] = {"lr": self.lr}
        if self.optim_fn in [torch.optim.Adam, torch.optim.AdamW]:
            optim_kwargs["eps"] = self.eps
        if self.optim_fn == torch.optim.AdamW:
            optim_kwargs["weight_decay"] = self.weight_decay
        return self.optim_fn(self.parameters(), **optim_kwargs)


class MuLinearMLP(pl.LightningModule):
    """
    A Multi-Layer Perceptron (MLP) model with 3 layers in maximal update parameterization (μP).

    Args:
        width:
            Width of the hidden layer.
        num_classes:
            Number of classes.
        bias:
            If ``True``, use bias in the model.
        nonlin:
            Nonlinearity function to use in the model.
        optimizer:
            Name of the optimizer to use.
        input_mult:
            Input multiplier.
        output_mult:
            Output multiplier.
        loss_fn:
            Loss function to use.
        optim_fn:
            Optimizer to use.
        lr:
            Learning rate.
        eps:
            Epsilon value for the optimizer.
        weight_decay:
            Weight decay value for the optimizer.
    """

    def __init__(
        self,
        width: int = 128,
        num_classes: int = 10,
        bias: bool = False,
        nonlin: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        optimizer: Literal["sgd", "adam", "adamw"] = "sgd",
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
        optim_fn: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.width = width
        self.bias = bias
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

        self.fc_1 = MuLinear(
            in_features=3072,
            out_features=width,
            bias=bias,
            layer="input",
            optimizer=optimizer,
            weight_init_std=(1 / (math.sqrt(3072) * self.input_mult)),
            base_width=128,
        )
        self.fc_2 = MuLinear(
            in_features=width,
            out_features=width,
            bias=bias,
            layer="hidden",
            optimizer=optimizer,
            weight_init_std=(1 / math.sqrt(128)),
            base_width=128,
        )
        self.fc_3 = MuLinear(
            in_features=width,
            out_features=num_classes,
            bias=bias,
            layer="output",
            optimizer=optimizer,
            weight_init_std=0.0,
            base_width=128,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlin(self.fc_1(x) * self.input_mult)
        x = self.nonlin(self.fc_2(x))
        return self.fc_3(x) * self.output_mult

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data.view(data.size(0), -1))
        loss = self.loss_fn(output, target)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim_kwargs: dict[str, Any] = {"lr": self.lr}
        if self.optim_fn in [torch.optim.Adam, torch.optim.AdamW]:
            optim_kwargs["eps"] = self.eps
        if self.optim_fn == torch.optim.AdamW:
            optim_kwargs["weight_decay"] = self.weight_decay
        return self.optim_fn(self.parameters(), **optim_kwargs)


class CerebrasMLP(pl.LightningModule):
    """
    A Multi-Layer Perceptron (MLP) model with 3 layers in maximal update parameterization (μP).

    Args:
        width:
            Width of the hidden layer.
        num_classes:
            Number of classes.
        bias:
            If ``True``, use bias in the model.
        nonlin:
            Nonlinearity function to use in the model.
        optimizer:
            Name of the optimizer to use.
        input_mult:
            Input multiplier.
        output_mult:
            Output multiplier.
        loss_fn:
            Loss function to use.
        optim_fn:
            Optimizer to use.
        lr:
            Learning rate.
        eps:
            Epsilon value for the optimizer.
        weight_decay:
            Weight decay value for the optimizer.
    """

    def __init__(
        self,
        width: int = 128,
        num_classes: int = 10,
        bias: bool = False,
        nonlin: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
        optim_fn: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.width = width
        self.bias = bias
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

        self.fc_1 = nn.Linear(3072, width, bias=bias)
        self.fc_2 = nn.Linear(width, width, bias=bias)
        self.fc_3 = nn.Linear(width, num_classes, bias=bias)
        self.fc_1_initializer = {"name": "normal_", "mean": 0.0, "std": 1 / (math.sqrt(3072) * self.input_mult)}
        self.fc_2_initializer = {"name": "normal_", "mean": 0.0, "std": 1 / math.sqrt(128)}
        self.fc_3_initializer = {"name": "zeros_"}
        width_mult = width / 128
        scale_initializers_by_dimension(
            self.fc_2_initializer,
            width_scale=width_mult**-0.5,
        )
        self.output_mult /= width_mult
        self.width_mult = width_mult
        self.lr_adjustment_groups = {"fc_2": LRAdjustmentGroup("*fc_2*weight")}
        self.lr_adjustment_groups["fc_2"].set_scale(1 / width_mult)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        create_initializer(self.fc_1_initializer)(self.fc_1.weight)
        create_initializer(self.fc_2_initializer)(self.fc_2.weight)
        create_initializer(self.fc_3_initializer)(self.fc_3.weight)
        if self.bias:
            # zero biases
            nn.init.zeros_(self.fc_1.bias)
            nn.init.zeros_(self.fc_2.bias)
            nn.init.zeros_(self.fc_3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlin(self.fc_1(x) * self.input_mult)
        x = self.nonlin(self.fc_2(x))
        return self.fc_3(x) * self.output_mult

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data.view(data.size(0), -1))
        loss = self.loss_fn(output, target)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Group parameters by learning rate adjustment group
        params_groups_dict: dict[str, list[torch.Tensor]] = {}
        for name, param in self.named_parameters():
            for lr_group_name, lr_group in self.lr_adjustment_groups.items():
                if lr_group.param_filter(name):
                    params_groups_dict.setdefault(lr_group_name, []).append(param)
                    break
            else:
                params_groups_dict.setdefault("default", []).append(param)

        # Create parameter groups for the optimizer
        param_groups = []
        for lr_group_name, params in params_groups_dict.items():
            group_optim_kwargs = {"lr": self.lr, "eps": self.eps / self.width_mult}
            if self.optim_fn == torch.optim.AdamW:
                group_optim_kwargs["weight_decay"] = self.weight_decay
            if lr_group_name != "default":
                group_optim_kwargs["lr"] *= self.lr_adjustment_groups[lr_group_name].scale
                if self.optim_fn == torch.optim.AdamW:
                    # weight_decay is coupled with the learning rate in AdamW
                    # so we need to decouple it by scaling it inversely with the learning rate
                    # see https://github.com/microsoft/mup/issues/1
                    group_optim_kwargs["weight_decay"] /= self.lr_adjustment_groups[lr_group_name].scale
            param_groups.append({"params": params, **group_optim_kwargs})

        return self.optim_fn(param_groups, **{})


@pytest.mark.parametrize("implementation", ["sp", "mup_mu_linear", "mup_cerebras"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("nonlin", [F.relu])
@pytest.mark.parametrize(
    "optimizer,lr,input_mult,output_mult",
    [("sgd", 0.1, 2**-4, 2**5), ("adam", 0.01, 2**-3, 2**-4), ("adamw", 0.01, 2**-3, 2**-4)],
)
def test_mup(
    implementation: Literal["sp", "mup_mu_linear", "mup_cerebras"],
    train_loader: torch.utils.data.DataLoader,
    optimizer: Literal["sgd", "adam", "adamw"],
    lr: float,
    input_mult: float,
    output_mult: float,
    nonlin: Callable[[torch.Tensor], torch.Tensor],
    bias: bool,
):
    """
    Perform coordinate checking for the μP MLP model.


    Coordinate scaling of pre-activation layers, model weights, and changes in weights with respect to width
    are described in the Appendix J.2 of [1].

    **References:**

    1. `Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (Yang et al.)
       <https://arxiv.org/abs/2203.03466.pdf>`_.
    """
    if implementation == "mup_cerebras" and optimizer == "sgd":
        pytest.skip("Cerebras does not support SGD optimizer")

    nsteps = 3
    nseeds = 5
    widths = [2**i for i in range(7, 14)]
    df = get_coord_data_MLP(
        implementation=implementation,
        bias=bias,
        nonlin=nonlin,
        lr=lr,
        input_mult=input_mult,
        output_mult=output_mult,
        optim_name=optimizer,
        train_loader=train_loader,
        nsteps=nsteps,
        nseeds=nseeds,
        widths=widths,
    )
    if implementation == "sp":
        with pytest.raises(ValueError):
            coord_check_MLP(df, nsteps, bias, optimizer, atol=0.2)
    else:
        coord_check_MLP(df, nsteps, bias, optimizer, atol=0.1)
    gc.collect()
