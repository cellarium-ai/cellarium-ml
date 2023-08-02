# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod

import numpy as np
import pyro
import torch


class BaseModule(torch.nn.Module, metaclass=ABCMeta):
    """
    Base module for all scvi-distributed modules.
    """

    @staticmethod
    @abstractmethod
    def _get_fn_args_from_batch(
        tensor_dict: dict[str, np.ndarray | torch.Tensor]
    ) -> tuple[tuple, dict]:
        """
        Get forward method arguments from batch.
        """


class BasePredictModule(BaseModule):
    """
    Base module for all scvi-distributed modules that can be used for prediction.
    """

    @abstractmethod
    def predict(self, x_ng: torch.Tensor) -> torch.Tensor:
        """
        Perform prediction on data tensor.

        Args:
            x_ng: Data tensor.

        Returns:
            Prediction tensor.
        """


class PyroABCMeta(pyro.nn.module._PyroModuleMeta, ABCMeta):
    """
    Metaclass for Pyro modules.
    """


class BasePyroModule(pyro.nn.PyroModule, BaseModule, metaclass=PyroABCMeta):
    """
    Base module for all scvi-distributed Pyro modules.
    """


class BasePredictPyroModule(
    pyro.nn.PyroModule, BasePredictModule, metaclass=PyroABCMeta
):
    """
    Base module for all scvi-distributed Pyro modules that can be used for prediction.
    """
