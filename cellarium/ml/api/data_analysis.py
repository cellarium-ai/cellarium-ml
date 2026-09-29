# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

from cellarium.ml import CellariumAnnDataDataModule
from cellarium.ml.api.utils import get_h5ad_files_limits
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.utilities.data import AnnDataField, densify, to_float_tensor, to_torch_sparse_csr


def get_datamodule(
    h5ad_paths: list[str],
    obs_columns: dict[str, tuple[str, Callable]] = {},
    var_key: str | None = None,
    batch_size: int = 4096,
    shuffle: bool = True,
    train_size: float = 1.0,
    stage: Literal["fit", "validate", "predict", "test"] = "fit",
    nexus_extract_uniform_sizes: bool | None = None,
    num_workers: int = 0,
    accelerator: Literal["cpu", "cuda", "mps"] = "cpu",
):
    if nexus_extract_uniform_sizes is None:
        nexus_extract_uniform_sizes = all(["extract_files" in path for path in h5ad_paths])  # a guess

    datamodule = CellariumAnnDataDataModule(
        dadc=DistributedAnnDataCollection(
            filenames=h5ad_paths,
            limits=get_h5ad_files_limits(
                h5ad_paths,
                nexus_extract_uniform_sizes=nexus_extract_uniform_sizes,
            ),
            obs_columns_to_validate=[c[0] for c in obs_columns.values()],
            max_cache_size=2,
        ),
        batch_keys={
            "x_ng": AnnDataField(
                attr="X",
                convert_fn=to_torch_sparse_csr if accelerator not in ["mps"] else densify,  # type: ignore[arg-type]
            ),
            "var_names_g": AnnDataField(attr="var_names") if var_key is None else AnnDataField(attr="var", key=var_key),
            "obs_names_n": AnnDataField(attr="obs_names"),
            **{
                batch_key: AnnDataField(attr="obs", key=col, convert_fn=fn)
                for batch_key, (col, fn) in obs_columns.items()
            },
        },
        batch_size=batch_size,
        shuffle=shuffle,
        train_size=train_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    datamodule.setup(stage=stage)
    return datamodule


class ObsmMapping(dict):
    def __init__(self, n_obs, *args, **kwargs):
        self._n_obs = n_obs
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if len(value) != self._n_obs:
            raise ValueError(f"obsm['{key}'] has length {len(value)}, expected {self._n_obs} (n_obs)")
        super().__setitem__(key, value)


class CellariumData:
    def __init__(
        self,
        h5ad_paths: list[str],
        total_mrna_umis_column: str | None = None,
        var_key: str | None = None,
        obs_columns: dict[str, tuple[str, Callable]] = {},
        batch_size: int = 4096,
        shuffle: bool = True,
        train_size: float = 1.0,
        stage: Literal["fit", "validate", "predict", "test"] = "fit",
        nexus_extract_uniform_sizes: bool | None = None,
        datamodule_num_workers: int = 0,
        accelerator: Literal["cpu", "cuda", "mps"] = "cuda",
    ):
        if total_mrna_umis_column is not None:
            obs_columns["total_mrna_umis_n"] = (total_mrna_umis_column, to_float_tensor)
        self._datamodule = get_datamodule(
            h5ad_paths=h5ad_paths,
            obs_columns=obs_columns,
            var_key=var_key,
            batch_size=batch_size,
            shuffle=shuffle,
            train_size=train_size,
            stage=stage,
            nexus_extract_uniform_sizes=nexus_extract_uniform_sizes,
            num_workers=datamodule_num_workers,
            accelerator=accelerator,
        )
        self._obs = self._datamodule.dadc.obs.copy()
        self._obsm = ObsmMapping(n_obs=len(self._obs))
        self._var = self._datamodule.dadc.adatas[0].var.copy()
        self._hvg: pd.Series | None = None
        if "var_names_g" in self._datamodule.batch_keys:
            anndatafield: AnnDataField = self._datamodule.batch_keys["var_names_g"]
            if anndatafield.key is not None:
                self._var.set_index(anndatafield.key, inplace=True)

    @property
    def datamodule(self) -> CellariumAnnDataDataModule:
        return self._datamodule

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @property
    def obsm(self) -> ObsmMapping:
        return self._obsm

    @property
    def hvg(self) -> pd.Series | None:
        """Boolean HVG mask aligned to var_names_g, or None if not set."""
        return self._hvg

    @hvg.setter
    def hvg(self, value: pd.Series | np.ndarray | Sequence[str] | None) -> None:
        if value is None:
            self._hvg = None
            return
        var_names_g = self.datamodule.var_names_g
        if isinstance(value, pd.Series):
            mask = value.reindex(var_names_g)
            if mask.isna().any():
                raise ValueError(f"hvg is missing entries for: {mask[mask.isna()].index.tolist()[:5]}")
            self._hvg = mask.astype(bool)
        else:
            arr = np.asarray(value)
            if arr.dtype == bool:
                if len(arr) != len(var_names_g):
                    raise ValueError(f"hvg length {len(arr)} != n_vars {len(var_names_g)}")
                self._hvg = pd.Series(arr, index=var_names_g)
            else:
                unknown = set(arr) - set(var_names_g)
                if unknown:
                    raise ValueError(f"Unknown gene names in hvg: {sorted(unknown)[:5]}")
                self._hvg = pd.Series(np.isin(var_names_g, arr), index=var_names_g)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape [{len(self._obs)}, {len(self._var)}], "
            f"obsm keys: {list(self._obsm.keys())}, "
            f"hvg_set={self._hvg is not None})"
        )
