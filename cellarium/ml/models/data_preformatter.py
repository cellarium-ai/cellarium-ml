# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import torch

from cellarium.ml.models.model import CellariumModel, PredictMixin


class DataPreformatter(CellariumModel, PredictMixin):
    r"""
    Minimal pass-through model for the ``data_preformatter`` CLI workflow.

    Intended to be combined with
    :class:`~cellarium.ml.callbacks.PredictionWriterArrow` in a ``predict`` CLI
    run.  This model itself performs no transformations and writes nothing to
    disk.  All ``cpu_transforms`` and ``transforms`` upstream in the pipeline are
    applied by the dataloader / pipeline as usual, and
    :class:`~cellarium.ml.callbacks.PredictionWriterArrow` receives the fully
    accumulated post-transform batch via ``write_on_batch_end`` and writes it to
    Arrow IPC files.

    Both :meth:`predict` and :meth:`forward` return an empty dict so that
    :class:`~cellarium.ml.core.CellariumPipeline` accumulates all transform
    outputs into the final ``prediction`` dict, which the callback then receives.
    """

    def reset_parameters(self) -> None:
        # No parameters or buffers to reset.
        pass

    def forward(self, **kwargs: Any) -> dict:
        """Return ``{}``; the pipeline accumulates all transform outputs."""
        return {}

    def predict(self, **kwargs: Any) -> dict[str, np.ndarray | torch.Tensor]:
        """Return ``{}``; the pipeline accumulates all transform outputs."""
        return {}
