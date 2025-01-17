# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

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


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor with a bounded queue for task submissions.
    This class is used to prevent the queue from growing indefinitely when tasks are submitted,
    which can lead to an out-of-memory error.
    """

    def __init__(self, max_workers: int, max_queue_size: int):
        # Use a bounded queue for task submissions
        self._queue: Queue = Queue(max_queue_size)
        super().__init__(max_workers=max_workers)

    def submit(self, fn, /, *args, **kwargs):
        # Block if the queue is full to prevent task overload
        self._queue.put(None)
        future = super().submit(fn, *args, **kwargs)

        # When the task completes, remove a marker from the queue
        def done_callback(_):
            self._queue.get()

        future.add_done_callback(done_callback)
        return future


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
        max_threadpool_workers: int = 8,
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.prediction_size = prediction_size
        self.key = key
        self.executor = BoundedThreadPoolExecutor(
            max_workers=max_threadpool_workers,
            max_queue_size=max_threadpool_workers * 2,
        )
        self.gzip = gzip
        self.sufficient_disk_space_exists: bool | None = None

    def __del__(self):
        """Ensure the executor shuts down on object deletion."""
        self.executor.shutdown(wait=True)

    def check_disk_space(self, num_files: int | float) -> bool | None:
        """Check if there is enough disk space to write all predictions.

        Args:
            num_files:
                The total number of files to be written (num_predict_batches).

        Returns:
            bool | None:
                True if there is enough disk space to write all predictions, False otherwise.
                None if the first output file does not exist yet.
        """
        first_file_path = os.path.join(self.output_dir, "batch_0.csv" + (".gz" if self.gzip else ""))
        if not os.path.isfile(first_file_path):
            return None
        first_file_size = os.path.getsize(first_file_path)  # single file in bytes
        total_required_space = first_file_size * num_files  # total required space in bytes
        usage = shutil.disk_usage(self.output_dir)
        available_space = usage.free  # free space in bytes
        return total_required_space <= available_space

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

        # check output directory for sufficient disk space once
        if self.sufficient_disk_space_exists is None:
            self.sufficient_disk_space_exists = self.check_disk_space(num_files=trainer.num_predict_batches[0])
            if self.sufficient_disk_space_exists is False:
                raise RuntimeError(
                    f"Insufficient disk space at {self.output_dir} to write all predictions"
                )
