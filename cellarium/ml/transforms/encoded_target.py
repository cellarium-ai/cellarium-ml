# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from torch import nn

from cellarium.ml.data.fileio import read_pkl_from_gcs


class EncodedTargets(nn.Module):
    """
    Since the total y_categories is 2604 (more than present in the extract),
    we use the sorted list of total unique categories to find indices of y_n values
    and use these new indices as the targets.
    using as a transform so that unique cell types pickle file only has to be loaded once.
    """

    def __init__(
        self,
        unique_cell_types_nparray_path: str = 'gs://cellarium-file-system/curriculum/lrexp_human_training_split_20241106/models/shared_metadata/final_filtered_sorted_unique_cells_lrexp_human.pkl',
    ) -> None:
        super().__init__()
        self.unique_cell_types_nparray = read_pkl_from_gcs(unique_cell_types_nparray_path)


    def forward(
        self,y_n: np.ndarray
    ) -> torch.tensor:
        """
        Since the total y_categories is 2604 (more than present in the extract),
        we use the sorted list of total unique categories to find indices of y_n values
        and use these new indices as the targets.

        """
        return({'y_n':torch.tensor(np.searchsorted(self.unique_cell_types_nparray, y_n))})
