# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.models import IncrementalPCA
from cellarium.ml.transforms import Densify, Filter

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`. The datamodule is shuffled, as for training.


def _expected_x(cdata: CellariumData, positions: np.ndarray) -> np.ndarray:
    return np.asarray(cdata.datamodule.dadc[positions].X.todense())


def _collect(batches) -> dict:
    positions, x, names = [], [], []
    for batch in batches:
        positions.append(batch["row_indices_n"].numpy())
        x.append(
            batch["x_ng"].to_dense().numpy()
            if batch["x_ng"].is_sparse_csr or batch["x_ng"].is_sparse
            else batch["x_ng"].numpy()
        )
        names.extend(batch["obs_names_n"])
    return {"positions": np.concatenate(positions), "x": np.concatenate(x), "names": np.array(names)}


def _module(genes: list[str]) -> CellariumModule:
    return CellariumModule(
        cpu_transforms=[Filter(filter_list=genes, ordering=True)],
        transforms=[Densify()],
        model=IncrementalPCA(var_names_g=np.array(genes), n_components=2, perform_mean_correction=True),
    )


def test_predict_batches_cover_every_cell_once_with_their_positions(cdata):
    result = _collect(cdata.predict_batches(batch_size=7))

    assert sorted(result["positions"]) == list(range(cdata.n_obs))
    # each row of the batches is the cell at its position
    order = np.argsort(result["positions"])
    assert np.array_equal(result["x"][order], _expected_x(cdata, np.arange(cdata.n_obs)))
    assert np.array_equal(result["names"][order], cdata.obs_names.to_numpy())


def test_predict_batches_do_not_follow_the_settings_of_the_datamodule(cdata):
    datamodule = cdata.datamodule
    datamodule.shuffle = True
    datamodule.drop_last_indices = True
    datamodule.drop_incomplete_batch = True
    datamodule.n_pred = 5  # the (training) predict_dataloader would read only these cells
    datamodule.iteration_strategy = "same_order"

    result = _collect(cdata.predict_batches(batch_size=7))
    assert sorted(result["positions"]) == list(range(cdata.n_obs))


def test_predict_batches_of_a_view_cover_the_cells_of_the_view(cdata):
    selected = np.array([1, 4, 16, 22, 30, 39])
    view = cdata[selected]

    result = _collect(view.predict_batches(batch_size=4))
    assert sorted(result["positions"]) == list(range(len(selected)))  # positions in the view
    order = np.argsort(result["positions"])
    assert np.array_equal(result["x"][order], _expected_x(cdata, selected))
    assert np.array_equal(result["names"][order], cdata.obs_names.to_numpy()[selected])


def test_predict_batches_without_a_module_apply_no_transforms(cdata):
    batch = next(iter(cdata.predict_batches(batch_size=7)))
    assert batch["x_ng"].shape[1] == len(cdata.var)  # every gene: no filter
    assert len(batch["var_names_g"]) == len(cdata.var)


def test_predict_batches_with_a_module_apply_its_transforms(cdata):
    genes = list(cdata.datamodule.var_names_g[:5])
    module = _module(genes)

    result = _collect(cdata.predict_batches(module, batch_size=7))
    order = np.argsort(result["positions"])
    assert result["x"].shape == (cdata.n_obs, 5)
    assert np.array_equal(result["x"][order], _expected_x(cdata, np.arange(cdata.n_obs))[:, :5])


def test_predict_batches_of_a_view_with_a_module(cdata):
    genes = list(cdata.datamodule.var_names_g[:5])
    selected = np.array([2, 3, 17, 25, 38])
    result = _collect(cdata[selected].predict_batches(_module(genes), batch_size=2))
    order = np.argsort(result["positions"])
    assert np.array_equal(result["x"][order], _expected_x(cdata, selected)[:, :5])


def test_predict_batches_leave_the_module_as_they_found_it_and_do_not_change_the_grad_mode(cdata):
    module = _module(list(cdata.datamodule.var_names_g[:5]))
    module.train()
    grad_enabled = torch.is_grad_enabled()

    for batch in cdata.predict_batches(module, batch_size=7):
        assert not module.training  # inference mode while iterating
        assert torch.is_grad_enabled() == grad_enabled  # not held across the yields
        assert not batch["x_ng"].requires_grad
    assert module.training


def test_an_abandoned_iteration_restores_the_module(cdata):
    module = _module(list(cdata.datamodule.var_names_g[:5]))
    module.train()

    batches = cdata.predict_batches(module, batch_size=7)
    next(batches)
    assert not module.training
    batches.close()
    assert module.training


def test_predict_batches_move_batches_to_the_device_but_keep_the_positions_on_the_cpu(cdata):
    module = _module(list(cdata.datamodule.var_names_g[:5]))
    batch = next(iter(cdata.predict_batches(module, device="cpu", batch_size=7)))
    assert batch["x_ng"].device.type == "cpu"
    assert batch["row_indices_n"].device.type == "cpu"
    assert batch["row_indices_n"].dtype == torch.int64


def test_predict_batches_with_workers(make_h5ad_files):
    cdata = CellariumData(
        h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30), datamodule_num_workers=2
    )

    result = _collect(cdata.predict_batches(batch_size=7))
    assert sorted(result["positions"]) == list(range(cdata.n_obs))
    # the batches do arrive out of order (each worker reads its own contiguous stretch), so the positions matter
    assert list(result["positions"]) != list(range(cdata.n_obs))
    order = np.argsort(result["positions"])
    assert np.array_equal(result["x"][order], _expected_x(cdata, np.arange(cdata.n_obs)))


def test_predict_batches_with_workers_apply_the_cpu_transforms_of_a_module(make_h5ad_files):
    cdata = CellariumData(
        h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30), datamodule_num_workers=2
    )
    genes = list(cdata.datamodule.var_names_g[:5])

    result = _collect(cdata.predict_batches(_module(genes), batch_size=7))
    order = np.argsort(result["positions"])
    assert np.array_equal(result["x"][order], _expected_x(cdata, np.arange(cdata.n_obs))[:, :5])
