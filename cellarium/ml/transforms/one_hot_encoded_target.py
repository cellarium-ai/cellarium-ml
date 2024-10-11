# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from torch import nn

from cellarium.ml.data.fileio import read_pkl_from_gcs


class OneHotEncodedTargets(nn.Module):
    """
    when called, assigns multilabel targets. All parents of the target cell type get assigned as targets. 
    """

    def __init__(
        self,
        unique_cell_types_nparray_path: str = 'gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_y_categories.pkl',
    ) -> None:
        super().__init__()
        self.unique_cell_types_nparray = read_pkl_from_gcs(unique_cell_types_nparray_path)


    def forward(
        self,y_n: np.ndarray
    ) -> torch.tensor:
        """

        """
        out_array = np.zeros((len(y_n), len(self.unique_cell_types_nparray)), dtype=int)
        print(f"NIMISH UNIQUE CELLS ARE {self.unique_cell_types_nparray}")
        for i, target_name in enumerate(y_n):
            target_index = np.where(self.unique_cell_types_nparray == 'CL_' + target_name[3:])[0][0] #get index of target cell type
            # Set the corresponding columns to 1
            out_array[i,[target_index]] = 1

        return {'y_n':torch.tensor(out_array)}
