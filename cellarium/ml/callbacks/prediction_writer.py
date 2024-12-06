# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Sequence
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from cellarium.ml.data.fileio import read_pkl_from_gcs


def write_prediction(
    prediction: torch.Tensor,
    logits: torch.Tensor,
    ids: np.ndarray,
    cell_type_names: np.ndarray,
    query_ids: np.ndarray,
    output_dir: Path | str,
    postfix: int | str,
    columns: np.ndarray,
) -> None:
    """
    Write prediction to a CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(prediction.cpu(), columns=columns)
    #df = pd.DataFrame(logits.cpu(), columns=columns)
    df.insert(0,"query_cell_type_names", cell_type_names)
    df.insert(0,"query_cell_id", query_ids)
    df.insert(0, "db_ids", ids)
    output_path = os.path.join(output_dir, f"batch_{postfix}_probs.csv")
    df.to_csv(output_path, header=True, index=False)


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

    def __init__(self, output_dir: Path | str, prediction_size: int | None = None) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.prediction_size = prediction_size

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
        #columns = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_filtered_sorted_unique_cells.pkl")
        columns = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/lrexp_human_training_split_20241106/models/shared_metadata/final_filtered_sorted_unique_cells_lrexp_human.pkl")
        pred = prediction["cell_type_probs_nc"]
        logits = prediction["y_logits_nc"]
        if self.prediction_size is not None:
            pred = pred[:, : self.prediction_size]
            logits = logits[:,: self.prediction_size]
        #y_n = batch['y_n'].cpu().numpy()
        y_n = batch['y_n_predict'] # use for model variation 4 predictions when multiple classes are targets
        y_n_cell_type_ids = np.take(columns,y_n)
        #cell_type_names = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/ontology_term_id_to_cell_type_np_array.pkl")
        cell_type_names = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/lrexp_human_training_split_20241106/models/shared_metadata/ontology_term_id_to_cell_type_np_array_lrexp_human.pkl")
        y_n_cell_type_names = np.take(cell_type_names,y_n)
        write_prediction(
            prediction=pred,
            logits=logits,
            #ids=y_n_cell_type_ids,
            cell_type_names = y_n_cell_type_names,
            ids=batch["obs_names_n"],
            query_ids = y_n_cell_type_ids,
            output_dir=self.output_dir,
            postfix=batch_idx * trainer.world_size + trainer.global_rank,
            columns=columns[0:self.prediction_size],
        )