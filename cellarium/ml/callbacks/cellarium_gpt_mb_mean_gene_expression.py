# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils._pytree import tree_map


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
    # move to cpu using tree_map
    prediction = tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, prediction)
    df = pd.DataFrame(prediction)
    df.insert(0, "obs_names_n", obs_names_n)
    output_path = os.path.join(output_dir, f"batch_{postfix}.csv" + (".gz" if gzip else ""))
    to_csv_kwargs: dict[str, str | bool] = {"header": True, "index": False}
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
        :class:`~lightning.pytorch.Trainer` to ``False``. This is accomplished in the config
        file by including ``return_predictions: false`` at indent level 0. For example,

        .. code-block:: yaml

            trainer:
              ...
            model:
              ...
            data:
              ...
            return_predictions: false

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
        gzip: bool = True,
        max_threadpool_workers: int = 8,
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.executor = BoundedThreadPoolExecutor(
            max_workers=max_threadpool_workers,
            max_queue_size=max_threadpool_workers * 2,
        )
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
        gene_prompt_size_n = batch["gene_prompt_size_n"]
        gene_query_size_n = batch["gene_query_size_n"]
        total_mrna_umis_downsampled_n = batch["total_mrna_umis_downsampled_n"]

        label_nc_dict = batch["label_nc_dict"]
        logits_nck_dict = prediction
        label_weight_nc_dict = batch["label_weight_nc_dict"]

        loss_n_dict = {}
        cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        mse_loss_fn = torch.nn.MSELoss(reduction="none")
        mae_loss_fn = torch.nn.L1Loss(reduction="none")
        # Make sure that label_nc_dict is created by concatenating the gene_value and metadata labels
        # in the same order as the embeddings.
        key = "gene_value"
        label_nc = label_nc_dict[key]
        logits_nck = logits_nck_dict[key]
        assert isinstance(logits_nck, torch.Tensor)
        label_weight_nc = label_weight_nc_dict[key]
        assert isinstance(label_weight_nc, torch.Tensor)
        cross_entropy_loss_nc = cross_entropy_loss_fn(logits_nck.view(label_nc.numel(), -1), label_nc.view(-1).long()).reshape(label_nc.shape)
        loss_n_dict["cross_entropy"] = torch.sum(cross_entropy_loss_nc * label_weight_nc, dim=-1) / label_weight_nc.sum(dim=-1)

        # compute the mean gene_value from logits_nck expression in order to compute the mse and mae loss
        probs_nck = torch.softmax(logits_nck, dim=-1)
        k_range = torch.arange(logits_nck.shape[-1], device=logits_nck.device)
        mean_gene_value_nc = probs_nck @ k_range
        # compute the mse and mae loss
        mse_loss_nc = mse_loss_fn(mean_gene_value_nc, label_nc)
        mae_loss_nc = mae_loss_fn(mean_gene_value_nc, label_nc)
        loss_n_dict["mse"] = torch.sum(mse_loss_nc * label_weight_nc, dim=-1) / label_weight_nc.sum(dim=-1)
        loss_n_dict["mae"] = torch.sum(mae_loss_nc * label_weight_nc, dim=-1) / label_weight_nc.sum(dim=-1)

        loss_n_dict["prompt_size"] = gene_prompt_size_n.cpu().int()
        loss_n_dict["query_size"] = gene_query_size_n.cpu().int()
        loss_n_dict["total_mrna_umis_downsampled"] = total_mrna_umis_downsampled_n.cpu()

        if "obs_names_n" not in batch.keys():
            raise ValueError(
                "PredictionWriter callback requires the batch_key 'obs_names_n'. Add this to the YAML config."
            )
        assert isinstance(batch["obs_names_n"], np.ndarray)

        write_prediction(
            prediction=loss_n_dict,
            obs_names_n=batch["obs_names_n"],
            output_dir=self.output_dir,
            postfix=batch_idx * trainer.world_size + trainer.global_rank,
            gzip=self.gzip,
            executor=self.executor,
        )
