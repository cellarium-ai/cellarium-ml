# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pyro
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from scvid.module import ProbabilisticPCAPyroModule
from scvid.train import PyroTrainingPlan

n, g, k = 1000, 10, 2


class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"X": self.data[idx]}


@pytest.fixture
def x_ng():
    rng = torch.Generator()
    rng.manual_seed(1465)
    z_nk = torch.randn((n, k), generator=rng)
    w_kg = torch.randn((k, g), generator=rng)
    sigma = 0.6
    noise = sigma * torch.randn((n, g), generator=rng)
    x_ng = z_nk @ w_kg + noise
    return x_ng


@pytest.mark.parametrize(
    "ppca_flavor", ["marginalized", "diagonal_normal", "multivariate_normal"]
)
@pytest.mark.parametrize("learn_mean", [False, True])
@pytest.mark.parametrize("minibatch", [False, True], ids=["fullbatch", "minibatch"])
def test_probabilistic_pca(x_ng, minibatch, ppca_flavor, learn_mean):
    if learn_mean:
        x_mean_g = None
    else:
        x_mean_g = x_ng.mean(axis=0)

    # dataloader
    batch_size = n // 2 if minibatch else n
    train_loader = DataLoader(
        TestDataset(x_ng),
        batch_size=batch_size,
    )
    # model
    pyro.clear_param_store()
    ppca = ProbabilisticPCAPyroModule(
        n_cells=n, g_genes=g, k_components=k, ppca_flavor=ppca_flavor, mean_g=x_mean_g
    )
    training_plan = PyroTrainingPlan(ppca, optim_kwargs={"lr": 5e-2})
    # trainer
    trainer = pl.Trainer(accelerator="cpu", max_steps=1500)
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)

    # expected var
    expected_var = torch.var(x_ng, axis=0).sum()

    # actual var
    W_kg = ppca.W_kg.data
    sigma = ppca.sigma.data
    actual_var = (torch.diag(W_kg.T @ W_kg) + sigma**2).sum()

    np.testing.assert_allclose(expected_var, actual_var, rtol=0.05)

    # check that the inferred z has std of 1
    z = ppca.get_latent_representation(x_ng)

    np.testing.assert_allclose(z.std(), 1, rtol=0.04)
