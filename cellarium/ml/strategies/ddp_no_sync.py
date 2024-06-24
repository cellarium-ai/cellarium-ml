import logging
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.utilities.distributed import _distributed_is_initialized, _sync_ddp_if_available
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.types import ReduceOp
from lightning.pytorch.plugins import Precision
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import TBroadcast
from typing_extensions import override

log = logging.getLogger(__name__)


class DDPNoSyncStrategy(ParallelStrategy):
    """Strategy for training with multiple processes in parallel."""

    def __init__(
        self,
        accelerator: pl.accelerators.Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
    ):
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        log.debug(f"{self.__class__.__name__}: initializing DDP strategy")
        self._num_nodes = 1

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    # @property
    # def num_nodes(self) -> int:
    #     return self._num_nodes

    # @num_nodes.setter
    # def num_nodes(self, num_nodes: int) -> None:
    #     # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
    #     self._num_nodes = num_nodes

    # @property
    # def num_processes(self) -> int:
    #     return len(self.parallel_devices) if self.parallel_devices is not None else 0

    # @override
    # def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
    #     """Perform a all_gather on all processes."""
    #     return _all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)

    # @override
    # def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
    #     """Reduces a boolean decision over distributed processes. By default is analagous to ``all`` from the standard
    #     library, returning ``True`` only if all input decisions evaluate to ``True``. If ``all`` is set to ``False``,
    #     it behaves like ``any`` instead.

    #     Args:
    #         decision: A single input decision.
    #         all: Whether to logically emulate ``all`` or ``any``. Defaults to True.

    #     Returns:
    #         bool: The reduced boolean decision.

    #     """
    #     decision = torch.tensor(int(decision), device=self.root_device)
    #     decision = self.reduce(
    #         decision,
    #         reduce_op=ReduceOp.SUM,  # type: ignore[arg-type]
    #     )
    #     decision = bool(decision == self.world_size) if all else bool(decision)
    #     return decision

    # @contextmanager
    # def block_backward_sync(self) -> Generator:
    #     """Blocks ddp sync gradients behaviour on backwards pass.

    #     This is useful for skipping sync when accumulating gradients, reducing communication overhead
    #     Returns: context manager with sync behaviour off

    #     """
    #     if isinstance(self.model, pl.utilities.types.DistributedDataParallel):
    #         with self.model.no_sync():
    #             yield None
    #     else:
    #         yield None

    # @override
    # def teardown(self) -> None:
    #     assert self.cluster_environment is not None
    #     self.cluster_environment.teardown()
    #     super().teardown()

    def model_to_device(self) -> None:
        log.debug(f"{self.__class__.__name__}: moving model to device [{self.root_device}]...")
        assert self.model is not None
        self.model.to(self.root_device)

    def reduce(
        self,
        tensor: torch.Tensor | Any,
        group: Any | None = None,
        reduce_op: ReduceOp | str | None = "mean",
    ) -> torch.Tensor | Any:
        """Reduces the given tensor (e.g. across GPUs/processes).

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to reduce
            reduce_op: the reduction operation. Defaults to 'mean'.
                Can also be a string 'sum' or ReduceOp.

        """
        if isinstance(tensor, torch.Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def determine_ddp_device_ids(self) -> list[int] | None:
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    @override
    def barrier(self, name: str | None = None) -> None:
        if not _distributed_is_initialized():
            return

        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj_list = [obj]
        torch.distributed.broadcast_object_list(obj_list, src, group=_group.WORLD)
        return obj_list[0]

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("ddp_no_sync", cls, description=cls.__name__)
