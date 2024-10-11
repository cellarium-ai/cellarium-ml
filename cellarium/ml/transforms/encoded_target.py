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
        multilabel_flag: bool = True,
        child_parent_list_path: str = '/home/nmagre/P1_logistic_regression/child_parent_indices_list.pkl',
        unique_cell_types_nparray_path: str = 'gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_filtered_sorted_unique_cells.pkl',
    ) -> None:
        super().__init__()
        self.multilabel_flag = multilabel_flag
        self.child_parent_list = read_pkl_from_gcs(child_parent_list_path)
        self.unique_cell_types_nparray = read_pkl_from_gcs(unique_cell_types_nparray_path)


    def forward(
        self,y_n: np.ndarray
    ) -> dict[str, torch.tensor]:
        """

        """
        if self.multilabel_flag==0:
            return({'y_n':torch.tensor(np.searchsorted(self.unique_cell_types_nparray, y_n))})
        else:
            out_array = np.zeros((len(y_n), len(self.child_parent_list)), dtype=int)
            indices = np.searchsorted(self.unique_cell_types_nparray, y_n)
            for i, target_index in enumerate(indices):
                # Set the corresponding columns to 1
                out_array[i, [target_index]+self.child_parent_list[target_index]] = 1
            return {'y_n':torch.tensor(out_array)}
