# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

import torch


class BaseModule(torch.nn.Module, ABC):
    """
    Base module for all scvi-distributed modules.
    """

    @staticmethod
    @abstractmethod
    def _get_fn_args_from_batch(
        tensor_dict: dict[str, torch.Tensor]
    ) -> tuple[tuple, dict]:
        """
        Get forward method arguments from batch.
        """
