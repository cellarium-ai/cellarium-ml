import numpy as np
import pyro
import pytest
import torch
from pyro import infer, optim

from scvid.module import ProbabilisticPCAPyroModule

n, g, k = 1000, 10, 2


@pytest.fixture
def x_ng():
    torch.manual_seed(1465)
    z_nk = torch.randn(n, k)
    w_kg = torch.randn(k, g)
    sigma = 0.6
    noise = sigma * torch.randn(n, g)
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

    pyro.clear_param_store()
    ppca = ProbabilisticPCAPyroModule(
        n_cells=n, g_genes=g, k_components=k, ppca_flavor=ppca_flavor, mean_g=x_mean_g
    )
    elbo = infer.Trace_ELBO()
    adam = optim.Adam({"lr": 1e-2})
    svi = infer.SVI(ppca.model, ppca.guide, adam, elbo)
    for i in range(5000):
        if minibatch:
            batch_size = n // 2
            ind = torch.randint(n, (batch_size,))
            x_batch = x_ng[ind]
        else:
            x_batch = x_ng
        svi.step(x_batch)

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
