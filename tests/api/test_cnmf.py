# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from cellarium.ml.api import NMFOutput
from cellarium.ml.api._data_transforms import OnePassRecipe, fit_or_reuse
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.api.tools import cnmf, pca
from cellarium.ml.models import OnlineNonNegativeMatrixFactorization
from cellarium.ml.models.ols import StreamingOrdinaryLeastSquares
from cellarium.ml.transforms import NormalizeTotal, ZScore

from .conftest import _make_h5ad_files

os.environ["TORCH_COMPILE_DISABLE"] = "1"

N_CELLS, N_GENES, N_HVG = 100, 12, 8
K_VALUES = [2, 3]
N_REPLICATES = 8  # int(8 * 0.3) = 2 neighbors, the least that consensus needs


def make_cdata(tmp_path, n_files: int = 2, seed: int = 0, **kwargs) -> CellariumData:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = _make_h5ad_files(tmp_path, n_files=n_files, cells_per_file=N_CELLS // n_files, n_genes=N_GENES, seed=seed)
    return CellariumData(h5ad_paths=paths, batch_size=25, **kwargs)


def hvg_names(cdata: CellariumData) -> list[str]:
    return list(cdata.var.index[:N_HVG])


def train(cdata: CellariumData, k_values=K_VALUES, n_replicates=N_REPLICATES, **kwargs) -> NMFOutput:
    return cnmf(cdata, k_values, n_replicates=n_replicates, latent_dim=8, max_epochs=2, accelerator="cpu", **kwargs)


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> tuple[CellariumData, NMFOutput]:
    cdata = make_cdata(tmp_path_factory.mktemp("cnmf"))
    cdata.hvg = hvg_names(cdata)
    return cdata, train(cdata)


@pytest.fixture
def cdata(trained) -> CellariumData:
    cdata = trained[0]
    cdata.obsm.clear()
    return cdata


@pytest.fixture
def output(trained, cdata) -> NMFOutput:
    """A fresh output (with no consensus) for the trained module."""
    return NMFOutput.from_cdata(cdata)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def one_batch(output: NMFOutput) -> dict:
    """All the cells in one batch, preprocessed as the module does (the transforms run by hand, as a check)."""
    dadc = output.cdata.datamodule.dadc
    x_ng = torch.tensor(np.asarray(dadc[np.arange(output.cdata.n_obs)].X.todense()), dtype=torch.float32)
    batch = {"x_ng": x_ng, "var_names_g": np.asarray(output.cdata.datamodule.var_names_g)}
    assert output.nmf_module.pipeline is not None
    return output.nmf_module.pipeline[:-1](batch)


# ---- training ----


def test_cnmf_trains_registers_and_returns_an_output_bound_to_the_data(trained):
    cdata, output = trained

    assert isinstance(output, NMFOutput)
    assert output.cdata is cdata
    assert output.k_values == K_VALUES
    record = cdata.trained_modules["cnmf"]
    assert record.complete
    assert record.module is output.nmf_module
    assert record.config["k_values"] == K_VALUES
    assert record.config["n_replicates"] == N_REPLICATES
    assert record.config["std_statistics"] == "raw counts"
    for k in K_VALUES:
        assert output.model.factors_dict[k].shape == (N_REPLICATES, k, N_HVG)
    assert list(output.model.var_names_g) == hvg_names(cdata)


def test_cnmf_preprocesses_the_counts_like_cnmf(trained):
    """The HVGs on the cpu, then dense counts divided by the gene standard deviations: no normalization."""
    cdata, output = trained
    module = output.nmf_module
    assert [type(t).__name__ for t in module.cpu_transforms] == ["Filter"]
    assert [type(t).__name__ for t in module.transforms] == ["Densify", "DivideByScale"]

    counts = np.asarray(cdata.datamodule.dadc[np.arange(cdata.n_obs)].X.todense())
    std_g = fit_or_reuse(cdata, OnePassRecipe(normalize_total=False), "cpu").module.model.std_g.numpy()
    batch = one_batch(output)
    expected = counts[:, :N_HVG] / (std_g[:N_HVG] + 1e-4)
    assert np.allclose(batch["x_ng"].numpy(), expected, rtol=1e-4)


def test_cnmf_warns_and_uses_all_genes_without_hvgs(tmp_path):
    cdata = make_cdata(tmp_path)
    with pytest.warns(UserWarning, match="hvg is not set"):
        output = train(cdata, k_values=2, n_replicates=2)
    assert len(output.model.var_names_g) == N_GENES
    assert output.k_values == [2]


def test_cnmf_reuses_the_statistics_it_has_fit(tmp_path, fits):
    cdata = make_cdata(tmp_path)
    cdata.hvg = hvg_names(cdata)
    train(cdata, k_values=2, n_replicates=2)
    train(cdata, k_values=3, n_replicates=2)
    assert len(fits) == 1  # the statistics of the raw counts, once


@pytest.mark.parametrize("k_values", [[], [0], [2, 2], [-1, 3], [2.5]])
def test_cnmf_rejects_bad_k_values(tmp_path, k_values):
    cdata = make_cdata(tmp_path)
    with pytest.raises(ValueError, match="distinct positive integers"):
        cnmf(cdata, k_values)


def test_cnmf_writes_its_checkpoint_only_where_asked(tmp_path):
    cdata = make_cdata(tmp_path / "data")
    cdata.hvg = hvg_names(cdata)
    train(cdata, k_values=2, n_replicates=2, checkpoint_dir=str(tmp_path / "ckpt"))
    assert (tmp_path / "ckpt" / "NMF.ckpt").exists()


# ---- construction ----


def test_from_cdata_needs_the_trained_module(cdata):
    with pytest.raises(ValueError, match="No trained module 'missing'"):
        NMFOutput.from_cdata(cdata, key="missing")


def test_an_output_needs_an_nmf_module(tmp_path):
    cdata = make_cdata(tmp_path)
    cdata.hvg = hvg_names(cdata)
    pca(cdata, n_components=2, accelerator="cpu")
    with pytest.raises(ValueError, match="requires a module with a NonNegativeMatrixFactorization"):
        NMFOutput.from_cdata(cdata, key="pca")


def test_an_output_needs_the_genes_of_the_model_in_the_data(trained, tmp_path):
    _, output = trained
    other = make_cdata(tmp_path, n_files=2)
    other._var = other._var.rename(index={name: f"renamed_{name}" for name in other._var.index[:2]})
    with pytest.raises(ValueError, match="genes of the model are not in the data"):
        NMFOutput(output.nmf_module, other)


def test_repr_says_what_has_been_computed(output):
    assert "consensus for k=[]" in repr(output)
    assert "not computed" in repr(output)
    output.compute_consensus_factors()
    assert "consensus for k=[2, 3]" in repr(output)


# ---- consensus ----


def test_consensus_is_empty_until_computed(output):
    with pytest.raises(RuntimeError, match="compute_consensus_factors"):
        output.consensus
    with pytest.raises(RuntimeError, match="calculate_reconstruction_error"):
        output.reconstruction_error


def test_compute_consensus_factors_for_some_or_all_k(output):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    assert list(output.consensus) == [2]
    assert output.consensus[2]["consensus_D_kg"].shape == (2, N_HVG)
    output.compute_consensus_factors(k_values=[3])
    assert sorted(output.consensus) == [2, 3]
    assert output.consensus[2]["density_threshold"] == 1.0  # kept from before
    assert output.consensus[3]["density_threshold"] == 0.5  # the default of Kotliar
    output.compute_consensus_factors(density_threshold=1.0)  # all k
    assert all(result["density_threshold"] == 1.0 for result in output.consensus.values())


def test_compute_consensus_factors_does_not_touch_the_global_random_state(output):
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    output.compute_consensus_factors(density_threshold=1.0)
    assert torch.equal(torch.rand(1), expected)


def test_computing_a_consensus_forgets_the_reconstruction_error(output):
    output.compute_consensus_factors(density_threshold=1.0)
    output.calculate_reconstruction_error(accelerator="cpu")
    assert output._rec_error is not None
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    with pytest.raises(RuntimeError):
        output.reconstruction_error


def test_consensus_errors_name_the_k(output):
    with pytest.raises(ValueError, match="k=3: .*too few to cluster"):
        output.compute_consensus_factors(k_values=3, density_threshold=1e-6)
    with pytest.raises(ValueError, match="Invalid k value"):
        output.compute_consensus_factors(k_values=[2, 7])


# ---- passes over the cells ----


def test_reconstruction_error_of_the_k_that_have_a_consensus(output):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    rec_error = output.calculate_reconstruction_error(accelerator="cpu")
    assert list(rec_error) == [2]
    assert np.isfinite(rec_error[2]) and rec_error[2] > 0
    assert output.reconstruction_error is rec_error

    # the same as in one batch
    batch = one_batch(output)
    expected = output.model.reconstruction_error(batch["x_ng"], batch["var_names_g"], output.consensus)[2]
    assert rec_error[2] == pytest.approx(expected, rel=1e-3)


def test_reconstruction_error_needs_a_consensus(output):
    with pytest.raises(RuntimeError, match="compute_consensus_factors"):
        output.calculate_reconstruction_error(accelerator="cpu")


def test_the_k_selection_plot_covers_the_k_with_both(output):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    output.calculate_reconstruction_error(accelerator="cpu")
    output.compute_consensus_factors(k_values=3, density_threshold=1.0)  # forgets the error: needs a new one
    with pytest.raises(RuntimeError):
        output.k_selection_plot()
    output.calculate_reconstruction_error(accelerator="cpu")
    output.k_selection_plot()
    assert len(plt.get_fignums()) == 1


def test_loadings_are_stored_in_obsm_in_the_order_of_the_cells(output, cdata):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    assert output.compute_loadings(2, accelerator="cpu") is None

    stored = cdata.obsm["X_cnmf_k2"]
    assert stored.shape == (cdata.n_obs, 2) and stored.dtype == np.float32
    assert (stored >= 0).all()
    assert np.allclose(stored.sum(axis=1), 1.0, atol=1e-4) or (stored.sum(axis=1) == 0).any()  # normalized

    batch = one_batch(output)
    expected = output.model.infer_loadings(
        batch["x_ng"], batch["var_names_g"], output.consensus, k=2, normalize=True
    ).numpy()
    assert np.allclose(stored, expected, atol=2e-3)


def test_loadings_can_be_unnormalized_and_stored_under_another_key(output, cdata):
    output.compute_consensus_factors(k_values=[2, 3], density_threshold=1.0)
    output.compute_loadings(3, key_added="loadings_{k}", normalize=False, accelerator="cpu")
    assert list(cdata.obsm) == ["loadings_3"]
    batch = one_batch(output)
    expected = output.model.infer_loadings(
        batch["x_ng"], batch["var_names_g"], output.consensus, k=3, normalize=False
    ).numpy()
    assert np.allclose(cdata.obsm["loadings_3"], expected, atol=2e-3)


def test_loadings_go_to_disk_if_they_do_not_fit_and_are_the_same(output, cdata):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    output.compute_loadings(2, key_added="in_memory", accelerator="cpu")
    output.compute_loadings(2, key_added="on_disk", accelerator="cpu", max_gb=1e-9)

    assert type(cdata.obsm["in_memory"]) is np.ndarray
    assert isinstance(cdata.obsm["on_disk"], np.memmap) and not cdata.obsm["on_disk"].flags.writeable
    assert np.array_equal(cdata.obsm["in_memory"], cdata.obsm["on_disk"])


def test_loadings_use_the_max_gb_of_the_data(output, cdata):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    cdata.max_gb = 1e-9
    output.compute_loadings(2, accelerator="cpu")
    assert isinstance(cdata.obsm["X_cnmf_k2"], np.memmap)


def test_loadings_of_a_view_are_stored_in_the_view(output, cdata):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    output.compute_loadings(2, accelerator="cpu")
    positions = np.array([0, 7, 33, 61, 99])
    view = cdata[positions]
    output.compute_loadings(2, cdata=view, key_added="again", accelerator="cpu")

    assert "again" not in cdata.obsm
    assert view.obsm["again"].shape == (5, 2)
    assert np.allclose(view.obsm["again"], cdata.obsm["X_cnmf_k2"][positions], atol=2e-3)
    assert np.array_equal(view.obsm["X_cnmf_k2"], cdata.obsm["X_cnmf_k2"][positions])  # copied with the view


def test_an_interrupted_pass_stores_nothing(output, cdata, monkeypatch):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    calls = []
    infer_loadings = output.model.infer_loadings

    def interrupt_after_one_batch(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise KeyboardInterrupt
        return infer_loadings(*args, **kwargs)

    monkeypatch.setattr(output.model, "infer_loadings", interrupt_after_one_batch)
    with pytest.raises(KeyboardInterrupt):
        output.compute_loadings(2, accelerator="cpu", max_gb=1e-9)
    assert "X_cnmf_k2" not in cdata.obsm


def test_loadings_need_the_consensus_of_k(output):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    with pytest.raises(KeyError, match="Missing consensus_factors key k=3"):
        output.compute_loadings(3, accelerator="cpu")


def _all_gene_tpm_stats(cdata):
    stats = fit_or_reuse(cdata, OnePassRecipe(target_count=1_000_000), "cpu").module.model
    return stats.mean_g, stats.std_g, stats.var_names_g


def test_refit_matches_one_big_batch(output, cdata):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    mean_g, std_g, var_names_g = _all_gene_tpm_stats(cdata)

    result = output.refit_consensus_factors_zscored_tpm_all_genes(
        2, mean_g=mean_g, std_g=std_g, var_names_g=var_names_g, accelerator="cpu"
    )

    # the same, with all the cells in a single update
    inference = one_batch(output)
    loadings_nk = output.model.infer_loadings(
        inference["x_ng"], inference["var_names_g"], output.consensus, k=2, normalize=False
    )
    counts = torch.tensor(np.asarray(cdata.datamodule.dadc[np.arange(cdata.n_obs)].X.todense()), dtype=torch.float32)
    batch = {"x_ng": counts, "var_names_g": np.asarray(var_names_g)}
    batch |= NormalizeTotal(target_count=1_000_000)(x_ng=batch["x_ng"])
    batch |= ZScore(mean_g=mean_g, std_g=std_g, var_names_g=var_names_g, eps=1e-12)(
        x_ng=batch["x_ng"], var_names_g=batch["var_names_g"]
    )
    solver = StreamingOrdinaryLeastSquares(
        var_names_g=np.arange(2).astype(str), n_targets=N_GENES, univariate=False, ridge_penalty=0.0
    )
    solver.update(x_ng=loadings_nk, y_nk=torch.as_tensor(batch["x_ng"]))
    expected = solver.solve()
    assert isinstance(expected, torch.Tensor)

    assert result["factors_kg"].shape == (2, N_GENES)
    assert list(result["var_names_g"]) == list(cdata.var.index)  # all genes, not only the HVGs
    assert torch.allclose(result["factors_kg"], expected, atol=1e-5, rtol=1e-4)


def test_refit_computes_the_tpm_statistics_by_default(output, cdata):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    mean_g, std_g, var_names_g = _all_gene_tpm_stats(cdata)
    explicit = output.refit_consensus_factors_zscored_tpm_all_genes(
        2, mean_g=mean_g, std_g=std_g, var_names_g=var_names_g, accelerator="cpu"
    )
    default = output.refit_consensus_factors_zscored_tpm_all_genes(2, accelerator="cpu")
    assert torch.allclose(default["factors_kg"], explicit["factors_kg"], atol=1e-4)


def test_refit_needs_all_of_the_statistics_or_none(output):
    output.compute_consensus_factors(k_values=2, density_threshold=1.0)
    with pytest.raises(ValueError, match="must all be provided"):
        output.refit_consensus_factors_zscored_tpm_all_genes(2, mean_g=torch.zeros(N_GENES), accelerator="cpu")


# ---- choosing k ----


def test_default_k_selection_uses_the_kotliar_defaults_for_every_k(output):
    output.default_k_selection(accelerator="cpu")
    assert sorted(output.consensus) == K_VALUES
    assert all(r["density_threshold"] == 0.5 and r["local_neighborhood_size"] == 0.3 for r in output.consensus.values())
    assert sorted(output.reconstruction_error) == K_VALUES

    output.default_k_selection_plot(accelerator="cpu")
    assert len(plt.get_fignums()) == 1


@pytest.mark.parametrize("search", ["fast", "exhaustive"])
def test_maximal_stability_k_selection_keeps_the_best_consensus_of_every_k(output, search):
    output.maximal_stability_k_selection(
        fast_or_exhaustive=search, density_threshold_limits=(0.05, 0.9), local_neighborhood_size=0.5, accelerator="cpu"
    )
    assert sorted(output.consensus) == K_VALUES
    for result in output.consensus.values():
        assert 0.05 <= result["density_threshold"] <= 0.9 + 1e-9
        assert result["local_neighborhood_size"] == 0.5
    assert sorted(output.reconstruction_error) == K_VALUES

    output.maximal_stability_k_selection_plot(
        fast_or_exhaustive=search, density_threshold_limits=(0.05, 0.9), local_neighborhood_size=0.5, accelerator="cpu"
    )
    assert len(plt.get_fignums()) == 1


def test_maximal_stability_k_selection_errors_name_the_k(output):
    with pytest.raises(ValueError, match="k=2: No density threshold"):
        output.maximal_stability_k_selection(
            fast_or_exhaustive="exhaustive", density_threshold_limits=(1e-6, 1e-5), accelerator="cpu"
        )


# ---- plots ----


def test_density_histograms_for_some_or_all_k(output):
    output.plot_density_histograms()
    assert len(plt.get_fignums()) == 2
    output.plot_density_histograms(k_values=3, local_neighborhood_size=0.5)
    assert len(plt.get_fignums()) == 3
    with pytest.raises(ValueError, match="too small"):
        output.plot_density_histograms(local_neighborhood_size=0.1)


def test_clustermap_recomputes_with_the_neighborhood_size_it_had(output):
    pytest.importorskip("seaborn")
    output.compute_consensus_factors(k_values=2, density_threshold=0.5, local_neighborhood_size=0.5)

    output.plot_clustermap(2, density_threshold=0.9)

    assert output.consensus[2]["density_threshold"] == 0.9
    assert output.consensus[2]["local_neighborhood_size"] == 0.5  # not reset to the default
    assert len(plt.get_fignums()) == 2  # the clustermap and the histogram


def test_clustermap_without_a_consensus_asks_for_a_threshold(output):
    pytest.importorskip("seaborn")
    with pytest.raises(RuntimeError, match="compute_consensus_factors"):
        output.plot_clustermap(2)
    output.plot_clustermap(2, density_threshold=0.9)  # computes it
    assert output.consensus[2]["local_neighborhood_size"] == 0.3


def test_clustermap_does_not_hold_the_distance_matrix(output):
    pytest.importorskip("seaborn")
    output.plot_clustermap([2, 3], density_threshold=0.9)
    assert not any(
        isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[0] == value.shape[1] and value.shape[0] > 3
        for result in output.consensus.values()
        for value in result.values()
    )


# ---- save and load ----


def test_save_and_load_round_trip(output, cdata, tmp_path):
    output.default_k_selection(accelerator="cpu")
    path = str(tmp_path / "output.pt")
    output.save(path)

    loaded = NMFOutput.load(path, cdata)

    assert loaded.cdata is cdata
    assert loaded.k_values == output.k_values
    for k in K_VALUES:
        assert torch.equal(loaded.model.factors_dict[k], output.model.factors_dict[k])
        for key, value in output.consensus[k].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(loaded.consensus[k][key], value)
            else:
                assert loaded.consensus[k][key] == value
    assert loaded.reconstruction_error == output.reconstruction_error

    # it does the same: the transforms came with it
    output.compute_loadings(2, key_added="original", accelerator="cpu")
    loaded.compute_loadings(2, key_added="loaded", accelerator="cpu")
    assert np.allclose(cdata.obsm["loaded"], cdata.obsm["original"], atol=1e-5)


def test_save_and_load_without_a_consensus(output, cdata, tmp_path):
    path = str(tmp_path / "output.pt")
    output.save(path)
    loaded = NMFOutput.load(path, cdata)
    assert loaded._consensus == {} and loaded._rec_error is None


def test_a_saved_output_does_not_hold_the_distance_matrix_or_the_data(output, cdata, tmp_path):
    output.default_k_selection(accelerator="cpu")
    path = tmp_path / "output.pt"
    output.save(str(path))
    state = torch.load(path, weights_only=False)
    assert "cdata" not in state
    for result in state["consensus"].values():
        assert "filtered_euclidean_distance_matrix" not in result


def test_load_rejects_an_unknown_format(output, cdata, tmp_path):
    path = str(tmp_path / "output.pt")
    output.save(path)
    state = torch.load(path, weights_only=False)
    state["format_version"] = 99
    torch.save(state, path)
    with pytest.raises(ValueError, match="format version 99"):
        NMFOutput.load(path, cdata)


def test_load_binds_the_data_it_is_given(output, trained, tmp_path):
    path = str(tmp_path / "output.pt")
    output.save(path)
    view = trained[0][np.arange(0, 50)]
    loaded = NMFOutput.load(path, view)
    assert loaded.cdata is view


# ---- checkpoints of runs from the command line ----


def test_from_checkpoints_of_a_run_matches_the_output_of_the_run(tmp_path):
    cdata = make_cdata(tmp_path / "data")
    cdata.hvg = hvg_names(cdata)
    output = train(cdata, k_values=2, checkpoint_dir=str(tmp_path / "run"))

    combined = NMFOutput.from_checkpoints([str(tmp_path / "run" / "NMF.ckpt")], cdata)

    assert torch.equal(combined.model.factors_dict[2], output.model.factors_dict[2])
    assert [type(t).__name__ for t in combined.nmf_module.cpu_transforms] == ["Filter"]  # the split is kept
    assert [type(t).__name__ for t in combined.nmf_module.transforms] == ["Densify", "DivideByScale"]
    for name, each in (("run", output), ("combined", combined)):
        each.compute_consensus_factors(density_threshold=1.0)
        each.compute_loadings(2, key_added=name, accelerator="cpu")
    assert torch.equal(output.consensus[2]["consensus_D_kg"], combined.consensus[2]["consensus_D_kg"])
    # the transforms survived the checkpoint: the same loadings
    assert np.allclose(cdata.obsm["run"], cdata.obsm["combined"], atol=1e-5)


def test_from_checkpoints_split_by_k(tmp_path):
    """Two checkpoints cover disjoint k values, and the combined model has the union, exactly."""
    cdata = make_cdata(tmp_path / "data")
    cdata.hvg = hvg_names(cdata)
    first = train(cdata, k_values=[2, 3], n_replicates=2, checkpoint_dir=str(tmp_path / "run1"))
    second = train(cdata, k_values=[4, 5], n_replicates=2, checkpoint_dir=str(tmp_path / "run2"))

    combined = NMFOutput.from_checkpoints(
        [str(tmp_path / "run1" / "NMF.ckpt"), str(tmp_path / "run2" / "NMF.ckpt")], cdata
    )

    model = combined.model
    assert isinstance(model, OnlineNonNegativeMatrixFactorization)
    assert combined.k_values == [2, 3, 4, 5]
    for source in (first, second):
        for k in source.k_values:
            assert torch.equal(model.factors_dict[k], source.model.factors_dict[k])
    # no loadings buffers allocated
    assert not any(hasattr(model, f"loadings_{k}_rnk") for k in model.k_values)


def test_from_checkpoints_split_by_r(tmp_path):
    """Two checkpoints share k but have different numbers of replicates: they are concatenated along dim 0."""
    cdata = make_cdata(tmp_path / "data")
    cdata.hvg = hvg_names(cdata)
    first = train(cdata, k_values=3, n_replicates=2, checkpoint_dir=str(tmp_path / "run1"))
    second = train(cdata, k_values=3, n_replicates=3, checkpoint_dir=str(tmp_path / "run2"))

    combined = NMFOutput.from_checkpoints(
        [str(tmp_path / "run1" / "NMF.ckpt"), str(tmp_path / "run2" / "NMF.ckpt")], cdata
    )

    factors = combined.model.factors_dict[3]
    assert factors.shape == (5, 3, N_HVG)
    assert torch.equal(factors, torch.cat([first.model.factors_dict[3], second.model.factors_dict[3]]))


def test_from_checkpoints_rejects_different_genes(tmp_path):
    cdata = make_cdata(tmp_path / "data")
    cdata.hvg = hvg_names(cdata)
    train(cdata, k_values=2, n_replicates=2, checkpoint_dir=str(tmp_path / "run1"))
    cdata.hvg = list(cdata.var.index[2 : N_HVG + 2])
    train(cdata, k_values=3, n_replicates=2, checkpoint_dir=str(tmp_path / "run2"))

    with pytest.raises(ValueError, match="var_names_g mismatch"):
        NMFOutput.from_checkpoints([str(tmp_path / "run1" / "NMF.ckpt"), str(tmp_path / "run2" / "NMF.ckpt")], cdata)


def test_from_checkpoints_needs_checkpoints(trained):
    with pytest.raises(ValueError, match="must not be empty"):
        NMFOutput.from_checkpoints([], trained[0])
