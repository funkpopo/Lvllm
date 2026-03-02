# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.lk_adapters import (
    get_lk_cpu_supported_quant_method_names,
    get_lk_gpu_prefill_supported_quant_method_names,
)
from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
    _resolve_gpu_prefill_handler,
    _should_use_lk_cpu_path,
)


def _make_quant_method(class_name: str):
    quant_cls = type(class_name, (), {})
    return quant_cls()


def _make_layer(
    quant_method_name: str,
    *,
    layer_mode: str,
    lk_ready: bool,
    use_ep: bool,
) -> FusedMoE:
    layer = FusedMoE.__new__(FusedMoE)
    layer.quant_method = _make_quant_method(quant_method_name)
    layer.layer_name = "model.layers.0.mlp"
    layer.use_ep = use_ep

    layer.is_gpu_resident_layer = layer_mode == "resident"
    layer.is_gpu_prefill_layer = layer_mode == "prefill"
    layer.is_cpu_layer = layer_mode == "cpu"

    layer.lk_moe = object() if lk_ready else None
    layer.lk_moe_config = object() if lk_ready else None

    layer.should_use_gpu_prefill = lambda hidden_states: layer.is_gpu_prefill_layer
    return layer


def test_lk_quant_registry_supports_expected_baseline_types():
    cpu_supported = set(get_lk_cpu_supported_quant_method_names())
    prefill_supported = set(get_lk_gpu_prefill_supported_quant_method_names())

    assert "UnquantizedFusedMoEMethod" in cpu_supported
    assert "Fp8MoEMethod" in cpu_supported
    assert "CompressedTensorsW8A8Fp8MoEMethod" in cpu_supported
    assert "CompressedTensorsW8A8Int8MoEMethod" in cpu_supported
    assert "CompressedTensorsWNA16MoEMethod" in cpu_supported
    assert "ExpertsInt8MoEMethod" in cpu_supported
    assert "GGUFMoEMethod" in cpu_supported

    assert "UnquantizedFusedMoEMethod" in prefill_supported
    assert "Fp8MoEMethod" in prefill_supported
    assert "CompressedTensorsW8A8Fp8MoEMethod" in prefill_supported
    assert "CompressedTensorsW8A8Int8MoEMethod" in prefill_supported
    assert "CompressedTensorsWNA16MoEMethod" in prefill_supported
    assert "ExpertsInt8MoEMethod" in prefill_supported
    assert "GGUFMoEMethod" not in prefill_supported


@pytest.mark.parametrize(
    "quant_method_name,is_lk_cpu_supported",
    [
        ("UnquantizedFusedMoEMethod", True),
        ("Fp8MoEMethod", True),
        ("CompressedTensorsWNA16MoEMethod", True),
        ("CompressedTensorsW8A8Int8MoEMethod", True),
        ("GGUFMoEMethod", True),
        # Mainstream upstream quant methods should safely fall back to vLLM
        # when LK CPU path is unavailable.
        ("ExpertsInt8MoEMethod", True),
        ("BitsAndBytesMoEMethod", False),
        ("GPTQMarlinMoEMethod", False),
    ],
)
@pytest.mark.parametrize("layer_mode", ["resident", "prefill", "cpu"])
@pytest.mark.parametrize("lk_ready", [False, True])
@pytest.mark.parametrize("use_ep", [False, True])
def test_should_use_lk_cpu_path_matrix(
    quant_method_name: str,
    is_lk_cpu_supported: bool,
    layer_mode: str,
    lk_ready: bool,
    use_ep: bool,
):
    layer = _make_layer(
        quant_method_name,
        layer_mode=layer_mode,
        lk_ready=lk_ready,
        use_ep=use_ep,
    )
    hidden_states = torch.zeros(4, 8)
    actual = _should_use_lk_cpu_path(layer, hidden_states)

    expected = layer_mode == "cpu" and is_lk_cpu_supported and lk_ready
    assert actual == expected


def test_unknown_quant_method_uses_vllm_fallback():
    layer = _make_layer(
        "ModelOptFp8MoEMethod",
        layer_mode="cpu",
        lk_ready=False,
        use_ep=False,
    )
    assert layer.supports_vllm_quant_fallback()


def test_gpu_prefill_unsupported_message_exposes_support_matrix():
    layer = _make_layer(
        "BitsAndBytesMoEMethod",
        layer_mode="prefill",
        lk_ready=True,
        use_ep=False,
    )

    message = layer.format_lk_gpu_prefill_unsupported_message()
    assert "BitsAndBytesMoEMethod" in message

    supported = get_lk_gpu_prefill_supported_quant_method_names()
    assert len(supported) > 0
    for quant_name in supported:
        assert quant_name in message


def test_resolve_gpu_prefill_handler_for_supported_quant_method():
    layer = _make_layer(
        "ExpertsInt8MoEMethod",
        layer_mode="prefill",
        lk_ready=True,
        use_ep=False,
    )

    handler = _resolve_gpu_prefill_handler(layer, action="prepare")
    assert callable(handler)


def test_resolve_gpu_prefill_handler_reports_supported_quant_methods():
    layer = _make_layer(
        "BitsAndBytesMoEMethod",
        layer_mode="prefill",
        lk_ready=True,
        use_ep=False,
    )

    with pytest.raises(ValueError) as exc_info:
        _resolve_gpu_prefill_handler(layer, action="prepare")

    assert "BitsAndBytesMoEMethod" in str(exc_info.value)
    assert "Supported quant_method(s)" in str(exc_info.value)
