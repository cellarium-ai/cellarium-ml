# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod

import pyro
import torch


class BaseModule(torch.nn.Module, metaclass=ABCMeta):
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

    def embed(self, x_ng: torch.Tensor) -> torch.Tensor | None:
        """
        Embed data.

        Args:
            x_ng: Data tensor.

        Returns:
            Embedding tensor.
        """


class PyroABCMeta(pyro.nn.module._PyroModuleMeta, ABCMeta):
    """
    Metaclass for Pyro modules.
    """


class BasePyroModule(pyro.nn.PyroModule, BaseModule, metaclass=PyroABCMeta):
    """
    Base module for all scvi-distributed Pyro modules.
    """
