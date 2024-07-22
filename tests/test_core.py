# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import lightning.pytorch as pl
import torch
import pytest

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.utilities.data import AnnDataField, densify
from cellarium.ml.transforms import before_to_device, Filter, Log1p, BeforeBatchTransferContext, AfterBatchTransferContext
from cellarium.ml.core.pipeline import CellariumPipeline
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


@pytest.mark.parametrize("filter_before_transfer", [False, True])
def test_batch_transfer_contexts(tmp_path: Path, filter_before_transfer: bool) -> None:
    # directly check the pipeline: do the context managers divide the pipeline as intended

    filter_transform = Filter(['ENSG00000187642', 'ENSG00000078808'])
    module = CellariumModule(
        transforms=[
            before_to_device(filter_transform) if filter_before_transfer else filter_transform,
            Log1p(),
        ],
        model=BoringModel(),
    )
    module.configure_model()

    full_pipeline = CellariumPipeline([m for m in module.pipeline])  # not the same reference
    print("full pipeline")
    print(full_pipeline)
    with BeforeBatchTransferContext(module):
        print("pre-device pipeline:")
        print(module.pipeline)
        if filter_before_transfer:
            assert str(module.pipeline) == str(CellariumPipeline([before_to_device(filter_transform)]))
        else:
            assert str(module.pipeline) == str(CellariumPipeline([]))
    with AfterBatchTransferContext(module):
        print("post-device pipeline:")
        print(module.pipeline)
        if filter_before_transfer:
            assert str(module.pipeline) == str(CellariumPipeline([Log1p(), BoringModel()]))
        else:
            assert str(module.pipeline) == str(CellariumPipeline([filter_transform, Log1p(), BoringModel()]))
    assert str(module.pipeline) == str(full_pipeline)  # did the context managers put things back


@pytest.mark.parametrize("filter_before_transfer", [False, True])
def test_transforms(tmp_path: Path, filter_before_transfer: bool) -> None:
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
    filter_transform = Filter(['ENSG00000078808', 'ENSG00000272106', 'ENSG00000162585'])
    transforms=[
        before_to_device(filter_transform) if filter_before_transfer else filter_transform,
        Log1p(),
    ]
    module = CellariumModule(transforms=transforms, model=BoringModel())
    module.configure_model()

    # ensure the transformed data is as expected, rather manually
    adata = datamodule.dadc.adatas[0].adata
    print(adata)
    x_ng = adata.X
    var_names_g = adata.var_names
    out = {"x_ng": torch.tensor(x_ng.todense()), "var_names_g": var_names_g}
    for t in transforms:
        ann = t.forward.__annotations__
        out |= t(**{k: v for k, v in out.items() if k in ann})
    expected_transformed_data = out["x_ng"]
    print(expected_transformed_data)

    transformed_data = module.pipeline.transform({"x_ng": torch.tensor(x_ng.todense()), "var_names_g": var_names_g})["x_ng"]
    print(transformed_data)
    torch.testing.assert_allclose(transformed_data, expected_transformed_data)

    # train
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
