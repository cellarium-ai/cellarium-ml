# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any

import pyro
import torch

from cellarium.ml.utilities.types import BatchDict


class CellariumModel(torch.nn.Module, metaclass=ABCMeta):
    """
    Base class for Cellarium ML compatible models.
    """

    __call__: Callable[..., BatchDict]


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
    def predict(self, *args: Any, **kwargs: Any) -> BatchDict:
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
