# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)

pytestmark = pytest.mark.cpu_test


def _make_layer(monkeypatch: pytest.MonkeyPatch) -> FusedMoE:
    layer = object.__new__(FusedMoE)
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.layer.is_lk_moe_quant_on_gpu",
        lambda: False,
    )
    return layer


def test_repack_expert_weight_tensor_block_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_layer(monkeypatch)
    monkeypatch.setattr(layer, "_get_lk_repack_chunk_size", lambda *args, **kwargs: 1)

    weight = torch.arange(3 * 4 * 6, dtype=torch.int8).reshape(3, 4, 6)
    scale = torch.tensor(
        [
            [[1.0, 0.5], [0.25, 2.0]],
            [[2.0, 1.5], [1.0, 0.75]],
            [[0.5, 1.25], [1.5, 1.0]],
        ],
        dtype=torch.float32,
    )
    group_shape = (2, 3)

    actual = layer._repack_expert_weight_tensor(
        weight,
        scale,
        torch.float16,
        FusedMoeWeightScaleSupported.BLOCK,
        group_shape,
    )

    expected = (
        weight.to(torch.float32).reshape(3, 2, 2, 2, 3)
        * scale.reshape(3, 2, 1, 2, 1)
    ).reshape(3, 4, 6).to(torch.float16)

    assert actual.device.type == "cpu"
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_repack_expert_weight_tensor_channel_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_layer(monkeypatch)
    monkeypatch.setattr(layer, "_get_lk_repack_chunk_size", lambda *args, **kwargs: 2)

    weight = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]],
            [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]],
        ],
        dtype=torch.float16,
    )
    scale = torch.tensor(
        [
            [[1.0], [0.5], [0.25]],
            [[2.0], [1.5], [1.0]],
            [[0.1], [0.2], [0.3]],
        ],
        dtype=torch.float32,
    )

    actual = layer._repack_expert_weight_tensor(
        weight,
        scale,
        torch.float32,
        FusedMoeWeightScaleSupported.CHANNEL,
    )
    expected = weight.to(torch.float32) * scale

    assert actual.device.type == "cpu"
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
