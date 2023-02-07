import numpy as np
import pyro
import pytest
import torch
from pyro import infer, optim

from scvid.module import ProbabilisticPCAPyroModule

n, p, k = 1000, 10, 2


@pytest.fixture
def x_np():
    torch.manual_seed(1465)
    z_nk = torch.randn(n, k)
    w_kp = torch.randn(k, p)
    sigma = 0.6
    noise = sigma * torch.randn(n, p)
    x_np = z_nk @ w_kp + noise
    return x_np


@pytest.mark.parametrize(
    "marginalize", [False, True], ids=["unmarginalized", "marginalized"]
)
@pytest.mark.parametrize("minibatch", [False, True], ids=["fullbatch", "minibatch"])
def test_probabilistic_pca(x_np, minibatch, marginalize):
    x_mean = x_np.mean(axis=0)

    pyro.clear_param_store()
    ppca = ProbabilisticPCAPyroModule(
        n_obs=n, n_vars=p, n_components=k, mean=x_mean, marginalize_z=marginalize
    )
    elbo = infer.Trace_ELBO()
    adam = optim.Adam({"lr": 1e-2})
    svi = infer.SVI(ppca.model, ppca.guide, adam, elbo)
    for i in range(5000):
        if minibatch:
            batch_size = n // 2
            ind = torch.randint(n, (batch_size,))
            x_batch = x_np[ind]
        else:
            x_batch = x_np
        svi.step(x_batch)

    # expected var
    expected_var = torch.var(x_np, axis=0).sum()

    # actual var
    W = pyro.param("probabilistic_pca.W").data
    sigma = pyro.param("probabilistic_pca.sigma").data
    actual_var = (torch.diag(W.T @ W) + sigma**2).sum()

    np.testing.assert_allclose(expected_var, actual_var, rtol=0.05)

    # check that the inferred z has std of 1
    z = ppca.get_latent_representation(x_np)

    np.testing.assert_allclose(z.std(), 1, rtol=0.04)
