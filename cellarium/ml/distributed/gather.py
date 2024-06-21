# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> tuple[torch.Tensor, ...]:  # type: ignore
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        all_grads = torch.stack(grads)
        dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
        return all_grads[dist.get_rank()]
