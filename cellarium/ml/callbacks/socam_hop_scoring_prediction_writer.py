# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from cellarium.ml.hop_scoring import calculate_hop_metrics_for_batch


def _to_numpy(value: np.ndarray | torch.Tensor | Sequence[Any]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_pickle(path: str) -> Any:
    return pd.read_pickle(path)


def _is_gcs_path(path: str | Path) -> bool:
    return str(path).startswith("gs://")


def _join_output_path(output_dir: str | Path, filename: str) -> str:
    output_dir_str = str(output_dir).rstrip("/")
    return f"{output_dir_str}/{filename}"


def _replace_strings(values: np.ndarray, old: str | None, new: str | None) -> np.ndarray:
    values = values.astype(str)
    if old is None or new is None:
        return values
    return np.char.replace(values, old, new)


class SOCAMHopScoringPredictionWriter(pl.callbacks.BasePredictionWriter):
    """Calculate hop-level SOCAM prediction metrics and write one CSV per prediction batch.

    This callback expects current :class:`cellarium.ml.models.SOCAM` prediction output, specifically
    ``prediction["cell_type_probs_nc"]``. It uses ``pl_module.model.active_cl_names`` as the prediction-column order.
    GCS output paths are supported through pandas/fsspec, so the runtime environment must include ``gcsfs``.
    """

    def __init__(
        self,
        output_dir: Path | str,
        co_resource_path: str,
        key: str = "cell_type_probs_nc",
        obs_names_key: str = "obs_names_n",
        ground_truth_key: str = "cl_names_n",
        num_hops: int = 4,
        gzip: bool = False,
        ontology_term_id_replace_from: str | None = ":",
        ontology_term_id_replace_to: str | None = "_",
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.co_resource_path = co_resource_path
        self.key = key
        self.obs_names_key = obs_names_key
        self.ground_truth_key = ground_truth_key
        self.num_hops = num_hops
        self.gzip = gzip
        self.ontology_term_id_replace_from = ontology_term_id_replace_from
        self.ontology_term_id_replace_to = ontology_term_id_replace_to
        self.co_resource = _load_pickle(co_resource_path)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: dict[str, torch.Tensor],
        batch_indices: Sequence[int] | None,
        batch: dict[str, np.ndarray | torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.key not in prediction:
            raise ValueError(f"Prediction output does not contain key '{self.key}'.")
        if self.obs_names_key not in batch:
            raise ValueError(f"Prediction batch does not contain key '{self.obs_names_key}'.")
        if self.ground_truth_key not in batch:
            raise ValueError(f"Prediction batch does not contain key '{self.ground_truth_key}'.")

        model = getattr(pl_module, "model", None)
        active_cl_names = getattr(model, "active_cl_names", None)
        if active_cl_names is None:
            raise ValueError("SOCAMHopScoringPredictionWriter requires `pl_module.model.active_cl_names`.")

        prediction_scores_nc = _to_numpy(prediction[self.key])
        query_cell_ids = _to_numpy(batch[self.obs_names_key]).astype(str)
        ground_truth_cl_names = _replace_strings(
            _to_numpy(batch[self.ground_truth_key]),
            self.ontology_term_id_replace_from,
            self.ontology_term_id_replace_to,
        )
        cell_type_ontology_term_ids = _replace_strings(
            np.asarray(active_cl_names),
            self.ontology_term_id_replace_from,
            self.ontology_term_id_replace_to,
        )

        metrics_df = calculate_hop_metrics_for_batch(
            query_cell_ids=query_cell_ids,
            prediction_scores_nc=prediction_scores_nc,
            ground_truth_cl_names=ground_truth_cl_names,
            cell_type_ontology_term_ids=cell_type_ontology_term_ids,
            co_resource=self.co_resource,
            num_hops=self.num_hops,
        )
        metrics_df.sort_values(by="query_cell_id", inplace=True)

        if not _is_gcs_path(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        postfix = batch_idx * trainer.world_size + trainer.global_rank
        filename = f"batch_{postfix}.csv" + (".gz" if self.gzip else "")
        output_path = _join_output_path(self.output_dir, filename)
        to_csv_kwargs: dict[str, str | bool] = {"header": True, "index": False}
        if self.gzip:
            to_csv_kwargs["compression"] = "gzip"
        metrics_df.to_csv(output_path, **to_csv_kwargs)
