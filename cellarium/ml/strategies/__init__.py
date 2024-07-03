import sys

from lightning.fabric.utilities.registry import _register_classes
from lightning.pytorch.strategies import StrategyRegistry
from lightning.pytorch.strategies.strategy import Strategy

from cellarium.ml.strategies.ddp_no_parameters import DDPNoParametersStrategy

_register_classes(StrategyRegistry, "register_strategies", sys.modules[__name__], Strategy)

__all__ = ["DDPNoParametersStrategy"]
