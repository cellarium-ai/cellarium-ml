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
    ids: np.ndarray,
    cell_type_names: np.ndarray,
    query_ids: np.ndarray,
    output_dir: Path | str,
    postfix: int | str,
    columns: np.ndarray,
) -> None:
    """
    Write prediction to a CSV file.

    Args:
        prediction:
            The prediction to write.
        ids:
            The IDs of the cells.
        output_dir:
            The directory to write the prediction to.
        postfix:
            A postfix to add to the CSV file name.
        columns:
            name of columns to save the csv file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(prediction.cpu(), columns=columns)
    df.insert(0,"query_cell_type_names", cell_type_names)
    df.insert(0,"query_cell_id", query_ids)
    df.insert(0, "db_ids", ids)
    output_path = os.path.join(output_dir, f"batch_{postfix}.csv")
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
        #x_ng = prediction["x_ng"]
        columns = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_filtered_sorted_unique_cells.pkl")
        #collated_predictions = collate_fn(prediction)
        x_ng = prediction["cell_type_probs_nc"]
        if self.prediction_size is not None:
            x_ng = x_ng[:, : self.prediction_size]

        #assert isinstance(batch["y_n_predict"], np.ndarray)
        #print(f"NIMISH BATCH Y_N TYPE IS {type(batch["y_n"])}")
        #y_n = batch['y_n'].cpu().numpy()
        y_n = batch['y_n_predict']
        #y_n = np.argmax(y_n)
        y_n_cell_type_ids = np.take(columns,y_n)
        cell_type_names = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/ontology_term_id_to_cell_type_np_array.pkl")
        y_n_cell_type_names = np.take(cell_type_names,y_n)
        #print(f"NIMISH BATCH KEYS ARE {batch.keys()}")
        #print(f"NIMISH OBS NAMES ARE {batch['obs_names_n']}")
        write_prediction(
            prediction=x_ng,
            #ids=y_n_cell_type_ids,
            cell_type_names = y_n_cell_type_names,
            ids=batch["obs_names_n"],
            query_ids = y_n_cell_type_ids,
            output_dir=self.output_dir,
            postfix=batch_idx * trainer.world_size + trainer.global_rank,
            columns=columns[0:self.prediction_size],
        )

# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

# import json
# import os
# from collections.abc import Sequence
# from pathlib import Path

# import lightning.pytorch as pl
# import numpy as np
# import pandas as pd
# import torch

# from cellarium.ml.data.fileio import read_pkl_from_gcs


# def write_prediction(
#     query_cell_ids: np.ndarray,
#     matches: np.ndarray,
#     total_weights: np.ndarray,
#     total_neighbors: np.ndarray,
#     total_neighbors_unrecognized: np.ndarray,
#     output_dir: Path | str,
#     postfix: int | str,
# ) -> None:
#     """
#     Write prediction to a CSV file.

#     Args:
#         prediction:
#             The prediction to write.
#         ids:
#             The IDs of the cells.
#         output_dir:
#             The directory to write the prediction to.
#         postfix:
#             A postfix to add to the CSV file name.
#         columns:
#             name of columns to save the csv file.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#     annotations = [{'query_cell_id':qid,
#                     'matches': mt,
#                     'total_weight': float(tw),
#                     'total_neighbors': int(tn),
#                     'total_neighbors_unrecognized': int(tnu)}
#                     for qid, mt, tw, tn, tnu in
#                     zip(query_cell_ids,matches,total_weights, total_neighbors, total_neighbors_unrecognized)
#                     ]
#     output_path = os.path.join(output_dir, f"batch_{postfix}.json")
#     with open(output_path, 'w') as json_file:
#         json.dump(annotations, json_file, indent=4)



# class PredictionWriter(pl.callbacks.BasePredictionWriter):
#     """
#     Write predictions to a CSV file. The CSV file will have the same number of rows as the
#     number of predictions, and the number of columns will be the same as the prediction size.
#     The first column will be the ID of each cell.

#     .. note::
#         To prevent an out-of-memory error, set the ``return_predictions`` argument of the
#         :class:`~lightning.pytorch.Trainer` to ``False``.

#     Args:
#         output_dir:
#             The directory to write the predictions to.
#         prediction_size:
#             The size of the prediction. If ``None``, the entire prediction will be
#             written. If not ``None``, only the first ``prediction_size`` columns will be written.
#     """

#     def __init__(self, output_dir: Path | str, prediction_size: int | None = None) -> None:
#         super().__init__(write_interval="batch")
#         self.output_dir = output_dir
#         self.prediction_size = prediction_size

#     def write_on_batch_end(
#         self,
#         trainer: pl.Trainer,
#         pl_module: pl.LightningModule,
#         prediction: dict[str, torch.Tensor],
#         batch_indices: Sequence[int] | None,
#         batch: dict[str, np.ndarray | torch.Tensor],
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:
#         #x_ng = prediction["x_ng"]
#         columns = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_filtered_sorted_unique_cells.pkl")
#         columns=columns[0:self.prediction_size]
#         #collated_predictions = collate_fn(prediction)
#         probability_outputs = prediction["cell_type_probs_nc"]
#         if self.prediction_size is not None:
#             probability_outputs = probability_outputs[:, : self.prediction_size]
#         probability_outputs = probability_outputs.cpu().numpy()
#         #y_n = batch['y_n'].cpu().numpy()
#         #y_n = np.argmax(y_n)
#         #y_n_cell_type_ids = np.take(columns,y_n)
#         cell_type_names = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/ontology_term_id_to_cell_type_np_array.pkl")
#         #y_n_cell_type_names = np.take(cell_type_names,y_n)
#         #unique_cell_type_names = np.take(cell_type_names,columns)
#         unique_cell_type_names = cell_type_names[0:self.prediction_size]
#         duplicated_columns = pd.DataFrame(np.tile(columns, (probability_outputs.shape[0], 1)))
#         duplicated_unique_names = pd.DataFrame(np.tile(unique_cell_type_names, (probability_outputs.shape[0], 1)))
#         scores = pd.DataFrame(probability_outputs)
#         #df_1 = pd.DataFrame(np.column_stack((probability_outputs, duplicated_columns, duplicated_unique_names)))
#         df = pd.concat([scores.stack(),duplicated_columns.stack(), duplicated_unique_names.stack()],axis=1)
#         df.columns = ['score', 'cell_type_ontology_term_id', 'cell_type']

#         matches = np.array(df.groupby(level=0).apply(lambda df: df.to_dict(orient='records')))
#         write_prediction(
#             query_cell_ids=batch["obs_names_n"],
#             matches=matches,
#             total_weights = probability_outputs.sum(axis=1),
#             total_neighbors = np.ones(len(matches)) * len(cell_type_names),
#             total_neighbors_unrecognized = np.zeros(len(matches)),
#             output_dir=self.output_dir,
#             postfix=batch_idx * trainer.world_size + trainer.global_rank,
#         )
