# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import lightning.pytorch as pl
import pytest
from anndata import AnnData

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.utilities.data import AnnDataField, densify
from tests.common import BoringModel


@pytest.mark.parametrize("batch_size", [50, None])
@pytest.mark.parametrize("adata", [True, False])
def test_datamodule(tmp_path: Path, batch_size: int | None, adata: bool) -> None:
    datamodule = CellariumAnnDataDataModule(
        DistributedAnnDataCollection(
            filenames="https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
            shard_size=100,
        ),
        batch_size=100,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
        },
    )
    module = CellariumModule(model=BoringModel())
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, default_root_dir=tmp_path)
    trainer.fit(module, datamodule)

    ckpt_path = str(tmp_path / "lightning_logs/version_0/checkpoints/epoch=0-step=1.ckpt")
    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if adata:
        kwargs["dadc"] = datamodule.dadc.adatas[0].adata
    loaded_datamodule = CellariumAnnDataDataModule.load_from_checkpoint(ckpt_path, **kwargs)

    assert loaded_datamodule.batch_keys == datamodule.batch_keys
    assert loaded_datamodule.batch_size == batch_size or datamodule.batch_size
    if adata:
        assert isinstance(loaded_datamodule.dadc, AnnData)
    else:
        assert isinstance(loaded_datamodule.dadc, DistributedAnnDataCollection)
        assert loaded_datamodule.dadc.filenames == datamodule.dadc.filenames
