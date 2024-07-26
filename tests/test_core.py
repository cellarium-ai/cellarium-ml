# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.transforms import Filter, Log1p
from cellarium.ml.utilities.core import train_val_split
from cellarium.ml.utilities.data import AnnDataField, densify
from tests.common import BoringModel


@pytest.mark.parametrize("batch_size", [50, None])
def test_datamodule(tmp_path: Path, batch_size: int | None) -> None:
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


@pytest.mark.parametrize("filter_before_transfer", [False, True])
def test_transform_integration(tmp_path: Path, filter_before_transfer: bool) -> None:
    batch_size = 100
    filename = "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad"
    datamodule = CellariumAnnDataDataModule(
        DistributedAnnDataCollection(
            filenames=filename,
            shard_size=100,
        ),
        batch_size=batch_size,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
            "var_names_g": AnnDataField(attr="var_names"),
        },
    )
    var_name_list = ["ENSG00000078808", "ENSG00000272106", "ENSG00000162585"]
    filter_transform = Filter(var_name_list)
    if filter_before_transfer:
        before_transforms: list[torch.nn.Module] = [filter_transform]
        after_transforms: list[torch.nn.Module] = [Log1p()]
    else:
        before_transforms = []
        after_transforms = [filter_transform, Log1p()]
    module = CellariumModule(
        before_batch_transfer_transforms=before_transforms,
        after_batch_transfer_transforms=after_transforms,
        model=BoringModel(),
    )
    module.configure_model()

    # ensure the transformed data is as expected, rather manually
    adata = datamodule.dadc.adatas[0].adata
    print(adata)
    x_ng = adata.X
    var_names_g = adata.var_names
    out = {"x_ng": torch.tensor(x_ng.todense()), "var_names_g": var_names_g}
    for t in before_transforms + after_transforms:
        ann = t.forward.__annotations__
        out |= t(**{k: v for k, v in out.items() if k in ann})
    expected_transformed_data = out["x_ng"]
    print(expected_transformed_data)
    assert expected_transformed_data.shape == (batch_size, len(var_name_list))

    # CellariumPipeline.transform()
    transformed_data = module.pipeline.transform({"x_ng": torch.tensor(x_ng.todense()), "var_names_g": var_names_g})[  # type: ignore[union-attr]
        "x_ng"
    ]
    print(transformed_data)
    torch.testing.assert_allclose(transformed_data, expected_transformed_data)

    # during training, when hooks are used to compute the transforms
    transformed_data_during_training = module.on_after_batch_transfer(
        batch=module.on_before_batch_transfer(
            batch={"x_ng": torch.tensor(x_ng.todense()), "var_names_g": var_names_g},
            batch_idx=0,
        ),
        batch_idx=0,
    )["x_ng"]
    print(transformed_data_during_training)
    torch.testing.assert_allclose(transformed_data_during_training, expected_transformed_data)

    # CellariumModule.forward()
    with pytest.raises(AttributeError):
        # BoringModel() has no predict method, which is what is called by module.forward
        module.forward({"x_ng": torch.tensor(x_ng.todense()), "var_names_g": var_names_g})

    # CellariumPipeline.forward()
    module.pipeline.forward({"x_ng": torch.tensor(x_ng.todense()), "var_names_g": var_names_g})  # type: ignore[union-attr]

    # ensure training runs
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, default_root_dir=tmp_path)
    trainer.fit(module, datamodule)
