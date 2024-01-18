# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

from cellarium.ml.models import MuLinear
from cellarium.ml.utilities.testing import assert_slope_equals, get_coord_data

optim_dict = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


# adapted from https://github.com/microsoft/mup/blob/main/examples/MLP/main.py
def coord_check_MLP(
    mup: bool,
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
        mup:
            If ``True``, use the μP MLP model. Otherwise, use the SP MLP model.
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

    def gen(w: int) -> Callable[[], nn.Module]:
        def f() -> nn.Module:
            model: nn.Module
            if mup:
                model = MuMLP(
                    width=w,
                    bias=bias,
                    nonlin=nonlin,
                    optimizer=optim_name,
                    input_mult=input_mult,
                    output_mult=output_mult,
                )
            else:
                model = MLP(width=w, bias=bias, nonlin=nonlin, input_mult=input_mult, output_mult=output_mult)
            return model

        return f

    models = {w: gen(w) for w in widths}
    optim_fn = optim_dict[optim_name]

    return get_coord_data(
        models,
        train_loader,
        loss_fn=F.cross_entropy,
        lr=lr,
        optim_fn=optim_fn,
        nseeds=nseeds,
        nsteps=nsteps,
    )


@pytest.fixture
def train_loader(tmp_path: Path) -> torch.utils.data.DataLoader:
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=tmp_path, train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
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
    """

    def __init__(
        self,
        width: int = 128,
        num_classes: int = 10,
        bias: bool = False,
        nonlin: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        input_mult: float = 1.0,
        output_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.nonlin = nonlin
        self.bias = bias
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(3072, width, bias=bias)
        self.fc_2 = nn.Linear(width, width, bias=bias)
        self.fc_3 = nn.Linear(width, num_classes, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in_1 = self.fc_1.weight.shape[1]
        nn.init.normal_(self.fc_1.weight, std=1 / math.sqrt(fan_in_1))  # 1 / sqrt(d)
        self.fc_1.weight.data /= self.input_mult**0.5
        fan_in_2 = self.fc_2.weight.shape[1]
        nn.init.normal_(self.fc_2.weight, std=1 / math.sqrt(fan_in_2))  # 1 / sqrt(n)
        nn.init.zeros_(self.fc_3.weight)  # zero readout
        if self.bias:
            # zero biases
            nn.init.zeros_(self.fc_1.bias)
            nn.init.zeros_(self.fc_2.bias)
            nn.init.zeros_(self.fc_3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        x = self.nonlin(self.fc_2(x))
        return self.fc_3(x) * self.output_mult


class MuMLP(nn.Module):
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
    ) -> None:
        super().__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = MuLinear(
            in_features=3072,
            out_features=width,
            bias=bias,
            layer="input",
            optimizer=optimizer,
            weight_init_std=(1 / math.sqrt(3072 * self.input_mult)),
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
        x = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        x = self.nonlin(self.fc_2(x))
        return self.fc_3(x) * self.output_mult


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("nonlin", [F.relu, F.tanh])
@pytest.mark.parametrize(
    "optimizer,lr,input_mult,output_mult",
    [("sgd", 0.1, 2**-8, 2**5), ("adam", 0.01, 2**-6, 2**-4), ("adamw", 0.01, 2**-6, 2**-4)],
)
def test_mup(
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
    nsteps = 3
    nseeds = 5
    widths = [2**i for i in range(7, 14)]
    df = coord_check_MLP(
        mup=True,
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
    atol = 0.1
    for t in range(nsteps):
        df_t = df.loc[df.t == t]
        for i in range(1, 4):
            df_i = df_t.loc[df_t.module.str.contains(str(i))]
            # output scaling: Θ(1)
            df_out = df_i.loc[df.type == "out"]
            out = df_out.groupby("width").l1.mean()
            if any(out):
                assert_slope_equals(out, 0, loglog=True, atol=atol)

            # W
            df_param = df_i.loc[df.type == "param"]
            W = df_param[df_param.module.str.contains("weight")].groupby("width").l1.mean()
            if i == 1:
                # input layer scaling: Θ(1)
                assert_slope_equals(W, 0, loglog=True, atol=atol)
            elif i == 2:
                # hidden layer scaling: Θ(1/sqrt(n))
                assert_slope_equals(W, -0.5, loglog=True, atol=atol)
            elif i == 3:
                # output layer scaling: Θ(1/n)
                if any(W):
                    assert_slope_equals(W, -1, loglog=True, atol=atol)

            # b scaling: Θ(1)
            if bias:
                b = df_param[df_param.module.str.contains("bias")].groupby("width").l1.mean()
                if any(b):
                    assert_slope_equals(b, 0, loglog=True, atol=atol)

            # ∆W
            df_delta = df_i.loc[df.type == "delta"]
            dW = df_delta[df_delta.module.str.contains("weight")].groupby("width").l1.mean()
            if i == 1:
                # input layer scaling: Θ(1)
                if any(dW):
                    assert_slope_equals(dW, 0, loglog=True, atol=atol)
            elif i == 2:
                # hidden layer scaling: Θ(1/n)
                if any(dW):
                    if optimizer == "adamw":
                        # additional scaling due weight decay: Θ(1/sqrt(n))
                        assert_slope_equals(dW, -0.75, loglog=True, atol=0.27)
                    else:
                        assert_slope_equals(dW, -1, loglog=True, atol=atol)
            elif i == 3:
                # output layer scaling: Θ(1/n)
                assert_slope_equals(dW, -1, loglog=True, atol=atol)

            # ∆b scaling: Θ(1)
            if bias:
                db = df_delta[df_delta.module.str.contains("bias")].groupby("width").l1.mean()
                if any(db):
                    assert_slope_equals(db, 0, loglog=True, atol=atol)

    with pytest.raises(AssertionError):
        df = coord_check_MLP(
            mup=False,
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
        atol = 0.2
        for t in range(nsteps):
            df_t = df.loc[df.t == t]
            for i in range(1, 4):
                df_i = df_t.loc[df_t.module.str.contains(str(i))]
                # output scaling: Θ(1)
                df_out = df_i.loc[df.type == "out"]
                out = df_out.groupby("width").l1.mean()
                if any(out):
                    assert_slope_equals(out, 0, loglog=True, atol=atol)

                # W
                df_param = df_i.loc[df.type == "param"]
                W = df_param[df_param.module.str.contains("weight")].groupby("width").l1.mean()
                if i == 1:
                    # input layer scaling: Θ(1)
                    assert_slope_equals(W, 0, loglog=True, atol=atol)
                elif i == 2:
                    # hidden layer scaling: Θ(1/sqrt(n))
                    assert_slope_equals(W, -0.5, loglog=True, atol=atol)
                elif i == 3:
                    # output layer scaling: Θ(1/n)
                    if any(W):
                        assert_slope_equals(W, -1, loglog=True, atol=atol)

                # b scaling: Θ(1)
                if bias:
                    b = df_param[df_param.module.str.contains("bias")].groupby("width").l1.mean()
                    if any(b):
                        assert_slope_equals(b, 0, loglog=True, atol=0.05)

                # ∆W
                df_delta = df_i.loc[df.type == "delta"]
                dW = df_delta[df_delta.module.str.contains("weight")].groupby("width").l1.mean()
                if i == 1:
                    # input layer scaling: Θ(1)
                    if any(dW):
                        assert_slope_equals(dW, 0, loglog=True, atol=atol)
                elif i == 2:
                    # hidden layer scaling: Θ(1/n)
                    if any(dW):
                        if optimizer == "adamw":
                            # additional scaling due weight decay: Θ(1/sqrt(n))
                            assert_slope_equals(dW, -0.75, loglog=True, atol=0.27)
                        else:
                            assert_slope_equals(dW, -1, loglog=True, atol=atol)
                elif i == 3:
                    # output layer scaling: Θ(1/n)
                    assert_slope_equals(dW, -1, loglog=True, atol=atol)

                # ∆b scaling: Θ(1)
                if bias:
                    db = df_delta[df_delta.module.str.contains("bias")].groupby("width").l1.mean()
                    if any(db):
                        assert_slope_equals(db, 0, loglog=True, atol=atol)
