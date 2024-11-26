# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import lightning.pytorch as pl
import pytest

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.models import CellariumModel, PredictMixin
from cellarium.ml.utilities.data import AnnDataField, densify


class BlankModel(CellariumModel, PredictMixin):
    """model that does nothing but throw exceptions when a hook method is called"""

    def __init__(self):
        super().__init__()
        self.hooks_called: list[str] = []

    def reset_parameters(self):
        self.hooks_called = []

    def forward(self, *args, **kwargs):
        return {}

    def predict(self, *args, **kwargs):
        return {}

    def on_train_epoch_start(self, *args, **kwargs):
        self.hooks_called.append("on_train_epoch_start")

    def on_train_batch_end(self, *args, **kwargs):
        self.hooks_called.append("on_train_batch_end")

    def on_train_epoch_end(self, *args, **kwargs):
        self.hooks_called.append("on_train_epoch_end")

    def on_predict_batch_end(self, *args, **kwargs):
        self.hooks_called.append("on_predict_batch_end")

    def on_predict_epoch_end(self, *args, **kwargs):
        self.hooks_called.append("on_predict_epoch_end")


@pytest.fixture(scope="function")
def training_run_setup(tmp_path: Path, accelerator: str = "cpu") -> dict:
    n_cells_in_shard = 100
    datamodule = CellariumAnnDataDataModule(
        DistributedAnnDataCollection(
            filenames="https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
            shard_size=n_cells_in_shard,
        ),
        batch_size=n_cells_in_shard // 2,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
            "obs_names_n": AnnDataField(attr="obs_names"),
        },
        train_size=1.0,
    )
    module = CellariumModule(model=BlankModel())
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=2, default_root_dir=tmp_path)
    return {"trainer": trainer, "module": module, "datamodule": datamodule}


def test_module_implemented_hooks(training_run_setup: dict) -> None:
    trainer = training_run_setup["trainer"]
    module = training_run_setup["module"]
    datamodule = training_run_setup["datamodule"]

    # ensure hooks are called as expected during fit
    trainer.fit(module, datamodule)
    print("hooks called by 2 epoch training run with 2 batches per epoch, in order:")
    print(module.model.hooks_called)
    expected_order_of_hooks = [
        "on_train_epoch_start",
        "on_train_batch_end",
        "on_train_batch_end",
        "on_train_epoch_end",
        "on_train_epoch_start",
        "on_train_batch_end",
        "on_train_batch_end",
        "on_train_epoch_end",
    ]
    assert module.model.hooks_called == expected_order_of_hooks

    # ensure hooks are called as expected during predict
    module.model.reset_parameters()
    datamodule.setup(stage="predict")
    trainer.predict(module, datamodule)
    print("hooks called by 2 batch predict run:")
    print(module.model.hooks_called)
    assert 0
