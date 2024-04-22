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
    def backward(ctx, *grads) -> torch.Tensor:
        grad_out = grads[dist.get_rank()].contiguous()
        dist.all_reduce(grad_out, op=dist.ReduceOp.SUM)
        return grad_out
