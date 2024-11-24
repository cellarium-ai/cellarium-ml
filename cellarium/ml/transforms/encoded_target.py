# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import torch
from torch import nn

from cellarium.ml.data.fileio import read_pkl_from_gcs


class EncodedTargets(nn.Module):
    """
    when called, assigns multilabel targets. All parents of the target cell type get assigned as targets.
    """

    def __init__(
        self,
        multilabel_flag: bool = False,
        target_row_ancestors_col_torch_tensor_path: str = 'gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/target_row_ancestors_col_torch_tensor.pkl',
        unique_cell_types_nparray_path: str = 'gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_filtered_sorted_unique_cells.pkl',
    ) -> None:
        super().__init__()
        self.multilabel_flag = multilabel_flag
        self.target_row_ancestors_col_torch_tensor = read_pkl_from_gcs(target_row_ancestors_col_torch_tensor_path)
        self.unique_cell_types_nparray = read_pkl_from_gcs(unique_cell_types_nparray_path)


    def forward(
        self,y_n: np.ndarray
    ) -> dict[str, torch.tensor]:
        """

        """
        if self.multilabel_flag==0:
            return({'y_n':torch.tensor(np.searchsorted(self.unique_cell_types_nparray, y_n))})
        else:
            indices = np.searchsorted(self.unique_cell_types_nparray, y_n)
            return {'y_n':self.target_row_ancestors_col_torch_tensor[indices]}
            #return {'y_n':self.target_row_ancestors_col_torch_tensor[indices], 'y_n_predict':indices} # only use for prediction of model 4
