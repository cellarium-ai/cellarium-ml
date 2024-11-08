# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from cellarium.ml.layers import MultiHeadAttention
from tests.common import USE_CUDA


@pytest.mark.skipif(not USE_CUDA, reason="requires_cuda")
def test_attention():
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
    causal_mask: torch.Tensor | BlockMask = torch.tril(torch.ones(n, c, c)).bool().cuda()
    outputs["torch"] = attention(x, x, x, causal_mask)

    for attention_backend in ["flex", "math", "mem_efficient"]:  # type: ignore[assignment]
        attention.attention_backend = attention_backend
        if attention_backend == "flex":

            def causal_mask_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            causal_mask = create_block_mask(causal_mask_mod, B=None, H=None, Q_LEN=c, KV_LEN=c)
        else:
            causal_mask = torch.tril(torch.ones(n, c, c)).bool().cuda()
        outputs[attention_backend] = attention(x, x, x, causal_mask)
        assert outputs[attention_backend].shape == (n, c, d_model)
        assert torch.allclose(outputs[attention_backend], outputs["torch"])
