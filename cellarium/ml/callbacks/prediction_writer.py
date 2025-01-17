# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch


def write_prediction(
    prediction: torch.Tensor,
    obs_names_n: np.ndarray,
    output_dir: Path | str,
    postfix: int | str,
    gzip: bool = True,
    executor: ThreadPoolExecutor | None = None,
) -> None:
    """
    Write prediction to a CSV file.

    Args:
        prediction:
            The prediction to write.
        obs_names_n:
            The IDs of the cells.
        output_dir:
            The directory to write the prediction to.
        postfix:
            A postfix to add to the CSV file name.
        gzip:
            Whether to compress the CSV file using gzip.
        executor:
            The executor used to write the prediction. If ``None``, no executor will be used.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(prediction.cpu())
    df.insert(0, "obs_names_n", obs_names_n)
    output_path = os.path.join(output_dir, f"batch_{postfix}.csv" + (".gz" if gzip else ""))
    to_csv_kwargs: dict[str, str | bool] = {"header": False, "index": False}
    if gzip:
        to_csv_kwargs |= {"compression": "gzip"}

    def _write_csv(frame: pd.DataFrame, path: str) -> None:
        frame.to_csv(path, **to_csv_kwargs)

    if executor is None:
        _write_csv(df, output_path)
    else:
        executor.submit(_write_csv, df, output_path)


class PredictionWriter(pl.callbacks.BasePredictionWriter):
    """
    Write predictions to a CSV file. The CSV file will have the same number of rows as the
    number of predictions, and the number of columns will be the same as the prediction size.
    The first column will be the ID of each cell.

    .. note::
        To prevent an out-of-memory error, set the ``return_predictions`` argument of the
        :class:`~lightning.pytorch.Trainer` to ``False``.

    Args:
        output_dir:
            The directory to write the predictions to.
        prediction_size:
            The size of the prediction. If ``None``, the entire prediction will be
            written. If not ``None``, only the first ``prediction_size`` columns will be written.
        key:
            PredictionWriter will write this key from the output of `predict()`.
        gzip:
            Whether to compress the CSV file using gzip.
        max_threadpool_workers:
            The maximum number of threads to use to write the predictions using a ThreadPoolExecutor.
    """

    def __init__(
        self,
        output_dir: Path | str,
        prediction_size: int | None = None,
        key: str = "x_ng",
        gzip: bool = True,
        max_threadpool_workers: int = 4,
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.prediction_size = prediction_size
        self.key = key
        self.executor = ThreadPoolExecutor(max_workers=max_threadpool_workers)
        self.gzip = gzip

    def __del__(self):
        """Ensure the executor shuts down on object deletion."""
        self.executor.shutdown(wait=True)

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
        if self.key not in batch.keys():
            raise ValueError(
                f"PredictionWriter callback specified the key '{self.key}' as the relevant output of `predict()`,"
                " but the key is not present. Specify a different key as an input argument to the callback, or"
                " modify the output keys of `predict()`."
            )
        prediction_np = prediction[self.key]
        if self.prediction_size is not None:
            prediction_np = prediction_np[:, : self.prediction_size]

        if "obs_names_n" not in batch.keys():
            raise ValueError(
                "PredictionWriter callback requires the batch_key 'obs_names_n'. Add this to the YAML config."
            )
        assert isinstance(batch["obs_names_n"], np.ndarray)
        write_prediction(
            prediction=prediction_np,
            obs_names_n=batch["obs_names_n"],
            output_dir=self.output_dir,
            postfix=batch_idx * trainer.world_size + trainer.global_rank,
            gzip=self.gzip,
            executor=self.executor,
        )
