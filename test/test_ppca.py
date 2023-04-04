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

from scvid.data import read_h5ad_file
from scvid.module import ProbabilisticPCAPyroModule
from scvid.train import PyroTrainingPlan
from scvid.transforms import ZScoreLog1pNormalize

k = 20


class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"X": self.data[idx]}


@pytest.fixture
def x_ng():
    url = "https://github.com/YosefLab/scVI-data/blob/master/hca_subsampled_20k.h5ad?raw=true"
    adata = read_h5ad_file(url)
    X = torch.as_tensor(adata.X.toarray())
    X = X[:5000, :10000]
    transform = ZScoreLog1pNormalize(
        mean_g=0, std_g=None, perform_scaling=False, target_count=10_000
    )
    return transform(X)
    #  n, g = 20000, 26662
    #  rng = torch.Generator()
    #  rng.manual_seed(1465)
    #  z_nk = torch.randn((n, k), generator=rng)
    #  w_kg = torch.randn((k, g), generator=rng)
    #  sigma = 5.0
    #  noise = sigma * torch.randn((n, g), generator=rng)
    #  x_ng = z_nk @ w_kg + noise
    #  return x_ng


@pytest.fixture
def pca_fit(x_ng):
    _x_ng = np.asarray(x_ng)
    pca = PCA(n_components=k)
    pca.fit(_x_ng)
    return pca


@pytest.mark.parametrize(
    "ppca_flavor", ["marginalized", "diagonal_normal", "multivariate_normal"][:1]
)
@pytest.mark.parametrize("learn_mean", [False, True][1:])
@pytest.mark.parametrize(
    "minibatch", [False, True][:1], ids=["fullbatch", "minibatch"][:1]
)
def test_probabilistic_pca_multi_device(
    x_ng, pca_fit, minibatch, ppca_flavor, learn_mean
):
    n, g = x_ng.shape
    devices = int(os.environ.get("TEST_DEVICES", "1"))
    if learn_mean:
        x_mean_g = None
    else:
        x_mean_g = x_ng.mean(axis=0)

    # expected var
    expected_var = torch.var(x_ng, axis=0).sum()

    # dataloader
    # batch_size = n // 2 if minibatch else n
    batch_size = n // 5
    # batch_size = 5000
    train_loader = DataLoader(
        TestDataset(x_ng),
        batch_size=batch_size,
        shuffle=True,
    )
    # model
    pyro.clear_param_store()
    w = torch.sqrt(0.5 * expected_var / (g * k)).item()
    s = torch.sqrt(0.5 * expected_var.sum() / g).item()
    ppca = ProbabilisticPCAPyroModule(
        n_cells=n,
        g_genes=g,
        k_components=k,
        ppca_flavor=ppca_flavor,
        mean_g=x_mean_g,
        W_init_scale=w,
        sigma_init_scale=s,
        total_variance=expected_var,
    )
    training_plan = PyroTrainingPlan(ppca, optim_kwargs={"lr": 0.01})
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    # trainer
    trainer = pl.Trainer(
        # barebones=True,
        accelerator="gpu",
        devices=devices,
        max_steps=50000,
        log_every_n_steps=10,
        strategy="auto",
        callbacks=[lr_monitor],
    )
    # fit
    trainer.fit(training_plan, train_dataloaders=train_loader)
    # trainer.fit(training_plan, train_dataloaders=train_loader, ckpt_path="lightning_logs/version_15/checkpoints/epoch=2999-step=15000.ckpt")
    import pdb;pdb.set_trace()

    # actual var
    actual_var = ppca.var_explained_W + ppca.var_explained_sigma

    np.testing.assert_allclose(expected_var, actual_var, rtol=0.02)

    np.testing.assert_allclose(pca_fit.explained_variance_, ppca.L_k, rtol=0.02)

    # check that the inferred z has std of 1
    z = ppca.get_latent_representation(x_ng)

    np.testing.assert_allclose(z.std(), 1, rtol=0.02)
