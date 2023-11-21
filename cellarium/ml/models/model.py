# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from pyro.nn.module import PyroParam, _unconstrain
from torch.distributions import transform_to


class CellariumModel(torch.nn.Module, metaclass=ABCMeta):
    """
    Base class for Cellarium ML compatible models.
    """

    def __init__(self) -> None:
        super(torch.nn.Module, self).__setattr__("_pyro_params", OrderedDict())
        super().__init__()

    __call__: Callable[..., torch.Tensor | None]

    @staticmethod
    @abstractmethod
    def _get_fn_args_from_batch(tensor_dict: dict[str, np.ndarray | torch.Tensor]) -> tuple[tuple, dict]:
        """
        Get forward method arguments from batch.
        """

    def __getattr__(self, name: str) -> Any:
        if "_pyro_params" in self.__dict__:
            _pyro_params = self.__dict__["_pyro_params"]
            if name in _pyro_params:
                constraint, event_dim = _pyro_params[name]
                unconstrained_value = getattr(self, name + "_unconstrained")
                constrained_value = transform_to(constraint)(unconstrained_value)
                return constrained_value

        return super().__getattr__(name)

    def __setattr__(self, name: str, value: torch.Tensor | torch.nn.Module | PyroParam) -> None:
        if isinstance(value, PyroParam):
            # Create a new PyroParam, overwriting any old value.
            try:
                delattr(self, name)
            except AttributeError:
                pass
            constrained_value, constraint, event_dim = value
            self._pyro_params[name] = constraint, event_dim
            unconstrained_value = _unconstrain(constrained_value, constraint)
            super().__setattr__(name + "_unconstrained", unconstrained_value)
            return

        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._pyro_params:
            delattr(self, name + "_unconstrained")
            del self._pyro_params[name]
        else:
            super().__delattr__(name)


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
