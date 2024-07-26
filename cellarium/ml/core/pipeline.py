# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

from cellarium.ml.models import CellariumModel, PredictMixin, ValidateMixin
from cellarium.ml.utilities.core import call_func_with_batch


class CellariumPipeline(torch.nn.Module):
    """
    A pipeline of modules. Modules are expected to return a dictionary. The input dictionary is sequentially passed to
    (piped through) each module and updated with its output dictionary.

    When used within :class:`cellarium.ml.core.CellariumModule`, the last module in the pipeline is expected to be
    a model (:class:`cellarium.ml.models.CellariumModel`) and any preceding modules are expected to be data transforms.

    Example:

        >>> from cellarium.ml import CellariumPipeline
        >>> from cellarium.ml.transforms import Filter, NormalizeTotal, Log1p
        >>> from cellarium.ml.models import IncrementalPCA
        >>> pipeline = CellariumPipeline(
        ...     before_batch_transfer_transforms=[Filter(filter_list=[f"gene_{i}" for i in range(20)])],
        ...     after_batch_transfer_transforms=[NormalizeTotal(), Log1p()],
        ...     model=IncrementalPCA(var_names_g=[f"gene_{i}" for i in range(20)], n_components=10),
        ... )
        >>> batch = {"x_ng": x_ng, "total_mrna_umis_n": total_mrna_umis_n, "var_names_g": var_names_g}
        >>> output = pipeline.forward(batch)  # or pipeline.predict(batch)

    Args:
        modules:
            Modules to be executed sequentially.
    """

    def __init__(
        self,
        before_batch_transfer_transforms: torch.nn.ModuleList | list[torch.nn.Module] = [],
        after_batch_transfer_transforms: torch.nn.ModuleList | list[torch.nn.Module] = [],
        model: CellariumModel | None = None,
    ) -> None:
        super().__init__()
        self._before_transfer_module_list = torch.nn.ModuleList(before_batch_transfer_transforms)
        self._after_transfer_module_list = torch.nn.ModuleList(after_batch_transfer_transforms)
        self._model: CellariumModel | None = model
        if (self._model is not None) and (not isinstance(self._model, CellariumModel)):
            raise TypeError(
                f"The CellariumPipeline 'model' must be an instance of {CellariumModel} (if not None). "
                f"Got {self._model}"
            )

    @property
    def before_transfer_transforms(self) -> torch.nn.ModuleList:
        """The transforms pipeline that happens before transfer to device"""
        return self._before_transfer_module_list

    @property
    def after_transfer_transforms(self) -> torch.nn.ModuleList:
        """The transforms pipeline that happens after transfer to device"""
        return self._after_transfer_module_list

    @property
    def model(self) -> CellariumModel | None:
        """The model"""
        return self._model

    def _get_model_as_module_list(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.model]) if (self.model is not None) else torch.nn.ModuleList([])

    def pipe(
        self,
        batch: dict[str, np.ndarray | torch.Tensor],
        module_key: str,
        method: str = "forward",
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Pipe a batch through (part of) the pipeline specified by module_key,
        choosing which method to call on the modules.

        Args:
            batch:
                The batch to pipe through the pipeline.
            module_key:
                The key of the module list to pipe the batch through.
                Must be in ['before_transfer', 'after_transfer', 'model'].
            method:
                The method to call on the modules. Default is 'forward'.

        Returns:
            The batch after being piped through the pipeline (affected keys are updated).
        """

        match module_key:
            case "before_transfer":
                module_list = self.before_transfer_transforms
            case "after_transfer":
                module_list = self.after_transfer_transforms
            case "model":
                module_list = self._get_model_as_module_list()
            case _:
                raise ValueError(
                    f"Invalid module_key: {module_key}. Must be in ['before_transfer', 'after_transfer', 'model']"
                )

        for module in module_list:
            batch |= call_func_with_batch(getattr(module, method), batch)

        return batch

    def _pipeline(
        self, batch: dict[str, np.ndarray | torch.Tensor], method: str
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Called internally by forward, predict, and validate, this method pipes the batch through the pipeline.
        """
        batch = self.transform(batch)

        for module in self._get_model_as_module_list():
            batch |= call_func_with_batch(getattr(module, method), batch)

        return batch

    def transform(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Pipes the batch through the transforms, putting each transform on the device where the batch is located.
        This may be necessary, since during training, the _before_transfer_module_list will often be on CPU
        while the rest will be on GPU.
        """
        for module_list in [self.before_transfer_transforms, self.after_transfer_transforms]:
            for module in module_list:
                if hasattr(module, "device"):
                    original_device = module.device
                    module.to(batch["x_ng"].device)  # type: ignore[union-attr]
                batch |= call_func_with_batch(module.forward, batch)
                if hasattr(module, "device"):
                    module.to(original_device)

        return batch

    def forward(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor | np.ndarray]:
        return self._pipeline(batch=batch, method="forward")

    def predict(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray | torch.Tensor]:
        if not isinstance(self.model, PredictMixin):
            raise TypeError(f"The CellariumPipeline 'model' must be an instance of {PredictMixin}. Got {self.model}")
        return self._pipeline(batch=batch, method="predict")  # type: ignore[unreachable]

    def validate(self, batch: dict[str, np.ndarray | torch.Tensor]) -> None:
        if not isinstance(self.model, ValidateMixin):
            raise TypeError(f"The CellariumPipeline 'model' must be an instance of {ValidateMixin}. Got {self.model}")
        return self._pipeline(batch=batch, method="validate")  # type: ignore[unreachable]
