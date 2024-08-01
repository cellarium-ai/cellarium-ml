# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import lightning.pytorch as pl
import pytest

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.transforms import Filter, Log1p, NormalizeTotal
from cellarium.ml.utilities.core import train_val_split
from cellarium.ml.utilities.data import AnnDataField, densify
from tests.common import USE_CUDA, BoringModel


@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not USE_CUDA, reason="requires_cuda"))],
)
@pytest.mark.parametrize(
    "cpu_transforms",
    [None, [Filter(filter_list=["ENSG00000187642", "ENSG00000078808"])]],
    ids=[None, "cpu_filter"],
)
@pytest.mark.parametrize("transforms", [None, [NormalizeTotal(), Log1p()]], ids=[None, "normalize_log"])
@pytest.mark.parametrize("num_workers", [0, 2], ids=lambda n: f"{n}workers")
def test_cpu_transforms(
    tmp_path: Path, accelerator: str, num_workers: int, cpu_transforms: list | None, transforms: list | None
) -> None:
    datamodule = CellariumAnnDataDataModule(
        DistributedAnnDataCollection(
            filenames="https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
            shard_size=100,
        ),
        batch_size=100,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
            "var_names_g": AnnDataField(attr="var_names"),
        },
        num_workers=num_workers,
        test_mode=False,
    )

    def _new_module():
        return CellariumModule(cpu_transforms=cpu_transforms, transforms=transforms, model=BoringModel())

    def _check_pipeline(module: CellariumModule):
        if cpu_transforms is not None:
            module_cpu_transforms = module.cpu_transforms
            for i, t in enumerate(cpu_transforms):
                assert isinstance(module_cpu_transforms[i], t.__class__)
        if transforms is not None:
            module_transforms = module.transforms
            for i, t in enumerate(transforms):
                assert isinstance(module_transforms[i], t.__class__)
        assert isinstance(module.model, BoringModel)

    # no lightning
    module = _new_module()
    module.configure_model()
    _check_pipeline(module)

    # lightning
    module = _new_module()
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_steps=1, default_root_dir=tmp_path)
    trainer.fit(module, datamodule)
    _check_pipeline(module)

    # ensure the data from the dataloader is filtered if appropriate
    for batch in datamodule.train_dataloader():
        x_ng = batch["x_ng"]
        if cpu_transforms is not None:
            assert x_ng.shape[1] == 2
        else:
            assert x_ng.shape[1] == 36601  # full number of genes in test dataset
        break


@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not USE_CUDA, reason="requires_cuda"))],
)
@pytest.mark.parametrize("batch_size", [50, None])
def test_datamodule(tmp_path: Path, batch_size: int | None, accelerator: str) -> None:
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
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_steps=1, default_root_dir=tmp_path)
    trainer.fit(module, datamodule)

    ckpt_path = str(tmp_path / "lightning_logs/version_0/checkpoints/epoch=0-step=1.ckpt")
    adata = datamodule.dadc.adatas[0].adata
    kwargs = {"dadc": adata}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    loaded_datamodule = CellariumAnnDataDataModule.load_from_checkpoint(ckpt_path, **kwargs)

    assert loaded_datamodule.batch_keys == datamodule.batch_keys
    assert loaded_datamodule.batch_size == batch_size or datamodule.batch_size
    assert loaded_datamodule.dadc is adata


@pytest.mark.parametrize(
    "n_samples, train_size, val_size, n_train_expected, n_val_expected",
    [
        (100, 0.8, 0.1, 80, 10),
        (100, 0.8, 0.2, 80, 20),
        (100, 80, 0.2, 80, 20),
        (100, 80, 20, 80, 20),
        (100, 80, None, 80, 20),
        (100, None, 20, 80, 20),
        (100, 0.8, None, 80, 20),
        (100, None, 0.2, 80, 20),
        (100, None, None, 100, 0),
    ],
)
def test_train_val_split(
    n_samples: int,
    train_size: float | int | None,
    val_size: float | int | None,
    n_train_expected: int,
    n_val_expected: int,
) -> None:
    n_train_actual, n_val_actual = train_val_split(n_samples, train_size, val_size)
    assert n_train_actual == n_train_expected
    assert n_val_actual == n_val_expected
