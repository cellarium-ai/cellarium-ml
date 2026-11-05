# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cellarium.ml import CellariumModule
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.models import IncrementalPCA
from cellarium.ml.transforms import STAGES, CellFilter, Densify, Dropout, Duplicate, Filter
from tests.common import BoringModel

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`.


def _expected_x(cdata: CellariumData, positions: np.ndarray) -> np.ndarray:
    return np.asarray(cdata.datamodule.dadc[positions].X.todense())


def _collect_x(batches) -> tuple[np.ndarray, np.ndarray]:
    positions, x = [], []
    for batch in batches:
        positions.append(batch["row_indices_n"].numpy())
        x.append(batch["x_ng"].numpy())
    return np.concatenate(positions), np.concatenate(x)


def test_datamodule_applies_the_cpu_transforms_active_in_the_stage_of_the_dataloader(cdata):
    datamodule = cdata.datamodule
    module = CellariumModule(cpu_transforms=[Duplicate(stages=["validate"])], model=BoringModel())
    module.configure_model()
    # as if the module were fit by a trainer: a stage is running, and the module is the lightning module
    datamodule.trainer = SimpleNamespace(state=SimpleNamespace(stage="fit"), lightning_module=module)
    batch = [{"x_ng": np.ones((3, 2), dtype=np.float32), "var_names_g": np.array(["a", "b"])}]

    n_rows = {stage: datamodule._effective_collate_fn(stage)(batch)["x_ng"].shape[0] for stage in STAGES}

    # Duplicate is active only in "validate" here, as stated by `stages`
    assert n_rows == {"train": 3, "validate": 6, "test": 3, "predict": 3}
    # the composed function has to be picklable, to be sent to dataloader workers
    pickle.loads(pickle.dumps(datamodule._effective_collate_fn("validate")))


def _module_with_inactive_transforms(genes: list[str]) -> CellariumModule:
    # a cell filter on the cpu that would drop every cell, and an augmentation on the "gpu"
    return CellariumModule(
        cpu_transforms=[Filter(filter_list=genes, ordering=True), CellFilter(min_count_per_cell=10**9)],
        transforms=[Densify(), Dropout(0.9, 0.99, 1.0)],
        model=IncrementalPCA(var_names_g=np.array(genes), n_components=2, perform_mean_correction=True),
    )


def test_predict_batches_skip_the_transforms_that_are_not_active_in_predict(cdata):
    genes = list(cdata.datamodule.var_names_g[:5])

    positions, x = _collect_x(cdata.predict_batches(_module_with_inactive_transforms(genes), batch_size=7))

    # every cell is covered (no cell filter) and untouched (no dropout)
    assert sorted(positions) == list(range(cdata.n_obs))
    assert np.array_equal(x[np.argsort(positions)], _expected_x(cdata, np.arange(cdata.n_obs))[:, :5])


def test_predict_batches_stages_override_makes_a_transform_active_in_predict(cdata):
    genes = list(cdata.datamodule.var_names_g[:5])
    module = CellariumModule(
        cpu_transforms=[Filter(filter_list=genes, ordering=True)],
        transforms=[Densify(), Dropout(0.9, 0.99, 1.0, stages=["predict"])],
        model=IncrementalPCA(var_names_g=np.array(genes), n_components=2, perform_mean_correction=True),
    )

    positions, x = _collect_x(cdata.predict_batches(module, batch_size=7))

    expected = _expected_x(cdata, np.arange(cdata.n_obs))[:, :5]
    assert sorted(positions) == list(range(cdata.n_obs))
    assert np.any(x[np.argsort(positions)] != expected)  # the dropout was applied


def test_predict_batches_with_workers_skip_the_cpu_transforms_that_are_not_active_in_predict(make_h5ad_files):
    cdata = CellariumData(
        h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30), datamodule_num_workers=2
    )
    genes = list(cdata.datamodule.var_names_g[:5])

    positions, x = _collect_x(cdata.predict_batches(_module_with_inactive_transforms(genes), batch_size=7))

    assert sorted(positions) == list(range(cdata.n_obs))
    assert np.array_equal(x[np.argsort(positions)], _expected_x(cdata, np.arange(cdata.n_obs))[:, :5])


@pytest.mark.parametrize("stage", STAGES)
def test_module_pipeline_without_trainer_applies_the_cpu_transforms_by_stage(stage):
    # outside of a trainer the module's pipeline has the cpu transforms too, and they follow the same stages
    module = CellariumModule(cpu_transforms=[Duplicate(stages=["validate"])], model=BoringModel())
    module.configure_model()
    out = module.pipeline[:-1].transform({"x_ng": torch.ones(3, 2)}, stage)  # type: ignore[arg-type]
    assert out["x_ng"].shape[0] == (6 if stage == "validate" else 3)
