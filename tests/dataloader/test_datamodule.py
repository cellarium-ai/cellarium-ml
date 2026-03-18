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

    def _check_transform_lists_match(iterable_modules1, iterable_modules2, message):
        if iterable_modules1 is None:
            iterable_modules1 = []
        if iterable_modules2 is None:
            iterable_modules2 = []
        for m1, m2 in zip(iterable_modules1, iterable_modules2):
            assert isinstance(m1, m2.__class__), message
        assert len(iterable_modules1) == len(iterable_modules2), "Transform lists different lengths"

    # no lightning
    print("Checking pipeline configuration when CellariumModule is not used with a lightning Trainer... ", end="")
    module = _new_module()
    module.configure_model()
    _check_pipeline(module)
    _check_transform_lists_match(
        (cpu_transforms if cpu_transforms else []) + (transforms if transforms else []),
        module.module_pipeline[:-1],
        "Transforms in CellariumModule.module_pipeline do not contain CPU transforms when used outside lightning",
    )
    print("✓")

    # lightning
    print("Checking pipeline configuration when CellariumModule is used after a lightning Trainer... ")
    module = _new_module()
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_steps=1, default_root_dir=tmp_path)
    trainer.fit(module, datamodule)
    _check_pipeline(module)
    _check_transform_lists_match(
        (cpu_transforms if cpu_transforms else []) + (transforms if transforms else []),
        module.module_pipeline[:-1],
        "Transforms in CellariumModule.module_pipeline are incorrect when used by lightning",
    )
    print("    ... ✓")

    # ensure the data from the dataloader is not filtered
    print("Checking that the data from the dataloader is not filtered outside of the trainer... ", end="")
    for batch in datamodule.train_dataloader():
        x_ng = batch["x_ng"]
        assert x_ng.shape[1] == 36601  # full number of genes in test dataset
        break
    print("✓")

    # ensure loading from a checkpoint manually results in the correct location of transforms
    print(
        "Checking whether we can load a module from a checkpoint using CellariumModule.load_from_checkpoint()... ",
        end="",
    )
    ckpt_path = str(tmp_path / "lightning_logs/version_0/checkpoints/epoch=0-step=1.ckpt")
    loaded_module = CellariumModule.load_from_checkpoint(ckpt_path)
    print("✓")

    print("Ensuring the pipeline is correctly configured after loading from checkpoint... ")
    print("\nExpected cpu_transforms")
    print(cpu_transforms)
    print("Loaded cpu_transforms")
    print(loaded_module.cpu_transforms)
    _check_transform_lists_match(
        cpu_transforms,
        loaded_module.cpu_transforms,
        "CPU transforms do not match after loading from checkpoint manually",
    )

    print("\nExpected transforms")
    print(transforms)
    print("Loaded transforms")
    print(loaded_module.transforms)
    _check_transform_lists_match(
        transforms,
        loaded_module.transforms,
        "Transforms do not match after loading from checkpoint manually",
    )

    print("\nFull loaded module ---------")
    print(loaded_module)
    assert loaded_module._cpu_transforms_in_module_pipeline, (
        "Upon manual loading, flag for CPU transforms should be True"
    )
    print("    ... ✓")

    # ensure loading from a checkpoint at lightning `fit` time results in the correct location of transforms
    module = _new_module()
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_steps=2, default_root_dir=tmp_path)
    print("Training one more step starting from a checkpoint...")
    trainer.fit(module, datamodule, ckpt_path=ckpt_path)
    print("    ... ✓")

    print("Ensuring the pipeline is correctly configured in the restarted Trainer... ")
    assert isinstance(trainer.model, CellariumModule)  # mypy requires this for the following print statements
    print("\nExpected cpu_transforms")
    print(cpu_transforms)
    print("Loaded cpu_transforms")
    print(trainer.model.cpu_transforms)
    _check_transform_lists_match(
        cpu_transforms,
        trainer.model.cpu_transforms,
        "CPU transforms do not match after loading from checkpoint using lightning Trainer.fit(ckpt)",
    )

    print("\nExpected transforms")
    print(transforms)
    print("Loaded transforms")
    print(trainer.model.transforms)
    _check_transform_lists_match(
        transforms,
        trainer.model.transforms,
        "Transforms do not match after loading from checkpoint using lightning Trainer.fit(ckpt)",
    )

    print("\nFull loaded module ---------")
    print(trainer.model)
    assert trainer.model._cpu_transforms_in_module_pipeline, (
        "After Trainer.fit() checkpoint restart, flag for CPU transforms should be True"
    )
    print("    ... ✓")


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
