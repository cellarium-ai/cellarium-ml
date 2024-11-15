# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cellarium.ml.models import CellariumModel


@dataclass
class LinkArguments:
    """
    Arguments for linking the value of a target argument to the values of one or more source arguments.

    Args:
        source:
            Key(s) from which the target value is derived.
        target:
            Key to where the value is set.
        compute_fn:
            Function to compute target value from source.
        apply_on:
            At what point to set target value, ``"parse"`` or ``"instantiate"``.
    """

    source: str | tuple[str, ...]
    target: str
    compute_fn: Callable | None = None
    apply_on: str = "instantiate"


class ModelRegistry(dict):
    """This class is a Registry that stores information about the Training Strategies.

    The Strategies are mapped to strings. These strings are names that identify
    a strategy, e.g., "deepspeed". It also returns Optional description and
    parameters to initialize the Strategy, which were defined durng the
    registration.

    The motivation for having a StrategyRegistry is to make it convenient
    for the Users to try different Strategies by passing just strings
    to the strategy flag to the Trainer.

    Example::

        @StrategyRegistry.register("lightning", description="Super fast", a=1, b=True)
        class LightningStrategy:
            def __init__(self, a, b):
                ...

        or

        StrategyRegistry.register("lightning", LightningStrategy, description="Super fast", a=1, b=True)

    """

    def register(
        self,
        name: str,
        model: CellariumModel | None = None,
        *,
        link_arguments_list: list[LinkArguments] | None = None,
        trainer_defaults: dict[str, Any] | None = None,
        override: bool = False,
    ) -> CellariumModel | Callable[[CellariumModel], CellariumModel]:
        """Registers a strategy mapped to a name and with required metadata.

        Args:
            name : the name that identifies a strategy, e.g. "deepspeed_stage_3"
            model : model class
            description : strategy description
            override : overrides the registered strategy, if True
            init_params: parameters to initialize the strategy

        """
        if name in self and not override:
            raise ValueError(f"'{name}' is already present in the registry. HINT: Use `override=True`.")

        def do_register(model: CellariumModel) -> CellariumModel:
            self[name] = {
                "model": model,
                "model_name": name,
                "link_arguments_list": link_arguments_list,
                "trainer_defaults": trainer_defaults,
            }
            return model

        if model is not None:
            return do_register(model)

        return do_register

    # @override
    # def get(self, name: str, default: Optional[Any] = None) -> Any:
    #     """Calls the registered strategy with the required parameters and returns the strategy object.

    #     Args:
    #         name (str): the name that identifies a strategy, e.g. "deepspeed_stage_3"

    #     """
    #     if name in self:
    #         data = self[name]
    #         return data["strategy"](**data["init_params"])

    #     if default is not None:
    #         return default

    #     err_msg = "'{}' not found in registry. Available names: {}"
    #     available_names = ", ".join(sorted(self.keys())) or "none"
    #     raise KeyError(err_msg.format(name, available_names))

    def available_strategies(self) -> list:
        """Returns a list of registered strategies."""
        return list(self.keys())

    def __str__(self) -> str:
        return "Registered Models: {}".format(", ".join(self.keys()))
