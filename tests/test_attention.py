# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from cellarium.ml.layers import MultiHeadAttention
from tests.common import USE_CUDA


@pytest.mark.skipif(not USE_CUDA, reason="requires_cuda")
@pytest.mark.parametrize("mask_type", ["causal", "random"])
def test_attention(mask_type: Literal["causal", "random"]):
    n = 8
    c = 10
    d_model = 64
    use_bias = False
    n_heads = 4
    dropout_p = 0.0
    attention_logits_scale = 1.0
    attention_backend: Literal["flex", "math", "mem_efficient", "torch"] = "torch"
    attention_softmax_fp32 = True
    Wqkv_initializer = {"name": "trunc_normal_", "std": 0.02}
    Wo_initializer = {"name": "trunc_normal_", "std": 0.02}

    attention = MultiHeadAttention(
        d_model,
        use_bias,
        n_heads,
        dropout_p,
        attention_logits_scale,
        attention_backend,
        attention_softmax_fp32,
        Wqkv_initializer,
        Wo_initializer,
    ).cuda()

    outputs = {}

    x = torch.randn(n, c, d_model).cuda()
    attention_mask: torch.Tensor | BlockMask
    if mask_type == "random":
        attention_mask = random_tensor = torch.randint(0, 2, (n, c, c)).bool().cuda()
    elif mask_type == "causal":
        attention_mask = torch.tril(torch.ones(n, c, c)).bool().cuda()
    outputs["torch"] = attention(x, x, x, attention_mask)

    for attention_backend in ["flex", "math", "mem_efficient"]:  # type: ignore[assignment]
        attention.attention_backend = attention_backend
        if attention_backend == "flex":
            if mask_type == "random":

                def random_mask_mod(b, h, q_idx, kv_idx):
                    return random_tensor[b, q_idx, kv_idx]

                attention_mask = create_block_mask(random_mask_mod, B=n, H=None, Q_LEN=c, KV_LEN=c, BLOCK_SIZE=c)

            elif mask_type == "causal":

                def causal_mask_mod(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx

                attention_mask = create_block_mask(causal_mask_mod, B=None, H=None, Q_LEN=c, KV_LEN=c, BLOCK_SIZE=c)

        else:
            if mask_type == "random":
                attention_mask = random_tensor
            else:
                attention_mask = torch.tril(torch.ones(n, c, c)).bool().cuda()
        outputs[attention_backend] = attention(x, x, x, attention_mask)
        assert outputs[attention_backend].shape == (n, c, d_model)
        assert torch.allclose(outputs[attention_backend], outputs["torch"])
