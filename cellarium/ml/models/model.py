# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pyro
import torch


class CellariumModel(torch.nn.Module, metaclass=ABCMeta):
    """
    Base class for Cellarium ML compatible models.
    """

    __call__: Callable[..., torch.Tensor | None]

    @staticmethod
    @abstractmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        """
        Get forward method arguments from batch.
        """


class PyroABCMeta(pyro.nn.module._PyroModuleMeta, ABCMeta):
    """
    Metaclass for Pyro modules.
    """


class CellariumPyroModel(pyro.nn.PyroModule, CellariumModel, metaclass=PyroABCMeta):
    """
    Base class for Cellarium ML compatible Pyro models.
    """


class PredictMixin(metaclass=ABCMeta):
    """
    Abstract mixin class for models that can perform prediction.
    """

    @abstractmethod
    def predict(self, x_ng: torch.Tensor, **kwargs: Any) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """
        Perform prediction on data tensor.

        Args:
            x_ng:
                Data tensor.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Prediction tensor or dictionary of prediction tensors.
        """
