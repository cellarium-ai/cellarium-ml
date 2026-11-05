# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import ClassVar, Literal, get_args

from torch import nn

Stage = Literal["train", "validate", "test", "predict"]
"""The stages of a :class:`~cellarium.ml.core.CellariumPipeline` that a transform can be active in."""

STAGES: tuple[Stage, ...] = get_args(Stage)


class CellariumTransform(nn.Module):
    """
    Base class for transforms that are active only in some stages of a
    :class:`~cellarium.ml.core.CellariumPipeline`.

    The stages are ``"train"`` (:meth:`CellariumPipeline.forward`), ``"validate"``, ``"test"`` and ``"predict"``.
    Each transform declares the stages in which it is active by default with :attr:`default_stages`,
    and the user can replace that set at instantiation with the ``stages`` argument.
    The pipeline skips a transform in the stages it is not active in. Transforms that are not
    :class:`CellariumTransform` instances (plain :class:`torch.nn.Module`) are active in every stage.

    The stage is chosen by the pipeline method that is called (and by the dataloader that applies
    the CPU transforms), not by :attr:`torch.nn.Module.training`, so it does not depend on whether
    a module was put in ``eval`` mode, nor on a :class:`lightning.pytorch.Trainer` being used.
    Calling a transform directly (outside of a pipeline) always applies it.

    Args:
        stages:
            The stages in which the transform is active. If ``None`` (default), the transform is
            active in the stages of :attr:`default_stages`. Otherwise this replaces :attr:`default_stages`.
    """

    default_stages: ClassVar[frozenset[Stage]] = frozenset(STAGES)
    # only set on the instance when overridden, so that instances created before this attribute existed
    # (such as unpickled ones) fall back to ``default_stages``
    _stages: frozenset[Stage] | None = None

    def __init__(self, stages: Sequence[Stage] | None = None) -> None:
        super().__init__()
        if stages is not None:
            if isinstance(stages, str):
                raise TypeError(f"`stages` must be a sequence of stage names, not a string. Got {stages!r}")
            unknown = set(stages) - set(STAGES)
            if unknown:
                raise ValueError(f"Unknown stages {sorted(unknown)}. Valid stages are {list(STAGES)}.")
            self._stages = frozenset(stages)

    @property
    def stages(self) -> frozenset[Stage]:
        """The stages in which the transform is active."""
        return self.default_stages if self._stages is None else self._stages

    def is_active(self, stage: Stage) -> bool:
        """Whether the transform is active in ``stage``."""
        return stage in self.stages
