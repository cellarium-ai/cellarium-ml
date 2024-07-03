from collections.abc import Callable
from datetime import timedelta
from typing import Any, Literal

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.pytorch.plugins import Precision
from lightning.pytorch.strategies.ddp import DDPStrategy
from typing_extensions import override


class DDPNoParametersStrategy(DDPStrategy):
    """Strategy for training with multiple processes in parallel for no-parameter models."""

    def __init__(
        self,
        accelerator: pl.accelerators.Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
    ):
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method=start_method,
        )

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        return super(DDPStrategy, self).setup(trainer)

    @override
    def _setup_model(self, model: torch.nn.Module) -> torch.nn.Module:  # type: ignore[override]
        if list(model.parameters()):
            raise ValueError(f"{self.__class__.__name__} does not support models with parameters.")
        return super(DDPStrategy, self)._setup_model(model)

    @override
    def _register_ddp_hooks(self) -> None:
        raise NotImplementedError("This method should not be called for this strategy.")

    @override
    def _enable_model_averaging(self) -> None:
        raise NotImplementedError("This method should not be called for this strategy.")

    @override
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: Callable[[], Any],
        model: pl.LightningModule | torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> Any:
        pass

    @override
    def configure_ddp(self) -> None:
        raise NotImplementedError("This method should not be called for this strategy.")

    @override
    def pre_backward(self, closure_loss: torch.Tensor) -> None:
        super(DDPStrategy, self).pre_backward(closure_loss)

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("ddp_no_parameters", cls, description=cls.__name__)
