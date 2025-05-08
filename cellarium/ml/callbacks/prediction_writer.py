# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Sequence
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import multiprocessing

from cellarium.ml.data.fileio import read_pkl_from_gcs
from cellarium.ml.hop_scoring.hop_score_calculation import calculate_metrics_for_cas_output_in_batches_csv


def write_prediction(
    prediction: torch.Tensor,
    ids: np.ndarray,
    query_ids: np.ndarray,
    output_dir: Path | str,
    columns: np.ndarray,
    postfix: int | str,
    co_resource_path: str,
    output_csv_path: str
) -> None:
    """
    Write prediction to a CSV file.
    """
    single_batch_df_copy = pd.DataFrame(prediction.cpu(), columns=columns)
    ground_truth_cl_names = query_ids
    cell_type_ontology_term_id_array = columns
    db_id_array = ids
    co_resource = read_pkl_from_gcs(co_resource_path)
    cas_out_csv = calculate_metrics_for_cas_output_in_batches_csv(
    db_id_array = db_id_array,
    ground_truth_cl_names=ground_truth_cl_names,
    single_batch_df_copy=single_batch_df_copy,
    cell_type_ontology_term_id_array=cell_type_ontology_term_id_array,
    co_resource=co_resource,
    num_hops=4,
    batch_size= int(len(ground_truth_cl_names)/multiprocessing.cpu_count()))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = str(output_csv_path)+str(postfix)+".csv"
    cas_out_csv.sort_values(by='query_cell_id', inplace=True)
    cas_out_csv.to_csv(output_path, header=True, index=False)

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
    """

    def __init__(
            self, 
            output_dir: Path | str,
            co_resource_path: str, 
            output_csv_path: str,
            csv_columns_path: str,
            ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.co_resource_path = co_resource_path
        self.output_csv_path = output_csv_path
        self.columns = np.char.replace(read_pkl_from_gcs(csv_columns_path),':',"_")

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
        pred = prediction["cell_type_probs_nc"]
        y_n = batch['y_n'].cpu().numpy()
        y_n_cell_type_ids = np.take(self.columns,y_n)

        write_prediction(
            prediction=pred,
            ids=batch["obs_names_n"],
            query_ids = y_n_cell_type_ids,
            output_dir=self.output_dir,
            columns=self.columns,
            postfix=batch_idx * trainer.world_size + trainer.global_rank,
            co_resource_path=self.co_resource_path,
            output_csv_path = self.output_csv_path
        )