# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from torch import nn

from cellarium.ml.data.fileio import read_pickle_from_gcs


class one_hot_encoded_targets(nn.Module):
    """
when called, assigns multilabel targets. All parents of the target cell type get assigned as targets. 
    """

    def __init__(
        self,
        multilabel_flag: bool = True,
        child_parent_list_path: str = '/home/nmagre/P1_logistic_regression/child_parent_indices_list.pkl',
        unique_cell_types_nparray_path: str = '/home/nmagre/P1_logistic_regression/unique_cell_types.pkl',
    ) -> None:
        super().__init__()
        self.multilabel_flag = multilabel_flag
        self.child_parent_list = read_pickle_from_gcs(child_parent_list_path)
        self.unique_cell_types_nparray = read_pickle_from_gcs(unique_cell_types_nparray_path)


    def forward(
        self,y_n: np.ndarray
    ) -> dict[str, np.ndarray]:
        """

        """
        out_array = np.zeros((len(y_n), len(self.child_parent_list)), dtype=int)
        for i, target_name in enumerate(y_n):
            #print(f"TARGET NAME IS {target_name}")
            target_index = np.where(self.unique_cell_types_nparray == 'CL_' + target_name[3:])[0][0] #get index of target cell type
            # Set the corresponding columns to 1
            if self.multilabel_flag==1:
                out_array[i, [target_index]+self.child_parent_list[target_index]] = 1
            else:
                out_array[i,[target_index]] = 1

        return {'y_n':torch.tensor(out_array)}
