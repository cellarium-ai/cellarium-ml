# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import pyro
import pytest
import pytorch_lightning as pl
import torch
from sklearn.decomposition import PCA
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
    rng = np.random.default_rng(1465)
    z_nk = rng.standard_normal(size=(n, k), dtype=np.float32)
    w_kg = rng.standard_normal(size=(k, g), dtype=np.float32)
    sigma = 0.6
    noise = sigma * rng.standard_normal(size=(n, g), dtype=np.float32)
    x_ng = z_nk @ w_kg + noise
    return x_ng


@pytest.fixture
def pca_fit(x_ng):
    pca = PCA(n_components=k)
    pca.fit(x_ng)
    return pca


@pytest.mark.parametrize("ppca_flavor", ["marginalized", "linear_vae"])
@pytest.mark.parametrize("learn_mean", [False, True])
@pytest.mark.parametrize("minibatch", [False, True], ids=["fullbatch", "minibatch"])
def test_probabilistic_pca_multi_device(
    x_ng, pca_fit, minibatch, ppca_flavor, learn_mean
):
    n, g = x_ng.shape
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    if learn_mean:
        x_mean_g = None
    else:
        x_mean_g = torch.as_tensor(x_ng.mean(axis=0))

    # dataloader
    batch_size = n // 2 if minibatch else n
    train_loader = DataLoader(
        TestDataset(x_ng),
        batch_size=batch_size,
        shuffle=True,
    )
    # model
    pyro.clear_param_store()
    total_var = np.var(x_ng, axis=0).sum()
    w = np.sqrt(0.5 * total_var / (g * k))
    s = np.sqrt(0.5 * total_var / g)
    ppca = ProbabilisticPCAPyroModule(
        n_cells=n,
        g_genes=g,
        k_components=k,
        ppca_flavor=ppca_flavor,
        mean_g=x_mean_g,
        W_init_scale=w,
        sigma_init_scale=s,
    )
    training_plan = PyroTrainingPlan(ppca, optim_kwargs={"lr": 0.05})
    # trainer
    trainer = pl.Trainer(
        barebones=True,
        accelerator="cpu",
        devices=devices,
        max_steps=1500,
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)

    # total variance
    expected_total_var = np.var(x_ng, axis=0).sum()
    actual_total_var = ppca.var_explained_W + ppca.var_explained_sigma
    np.testing.assert_allclose(expected_total_var, actual_total_var, rtol=0.01)

    # variance explained be each PC
    expected_explained_var = pca_fit.explained_variance_
    actual_explained_var = ppca.L_k
    np.testing.assert_allclose(expected_explained_var, actual_explained_var, rtol=0.01)

    # absolute cosine similarity between expected and actual PCs
    cos_sim = torch.abs(
        torch.nn.functional.cosine_similarity(
            ppca.U_gk.T, torch.as_tensor(pca_fit.components_)
        )
    )
    np.testing.assert_allclose(np.ones(k), cos_sim, rtol=0.01)

    # check that the inferred z has std of 1
    z = ppca.get_latent_representation(torch.as_tensor(x_ng))
    np.testing.assert_allclose(z.std(correction=0), 1, rtol=0.03)
