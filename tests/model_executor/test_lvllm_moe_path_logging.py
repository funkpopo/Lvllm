# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
    _maybe_log_moe_path_hit,
)

pytestmark = pytest.mark.cpu_test


def test_lvllm_moe_path_log_includes_request_batch(caplog_vllm):
    layer = SimpleNamespace(layer_name="model.layers.3.mlp.experts")
    vllm_config = VllmConfig()

    with (
        caplog_vllm.at_level(logging.DEBUG, logger="vllm"),
        set_forward_context(
            None,
            vllm_config=vllm_config,
            num_tokens=5,
            additional_kwargs={
                "lvllm_moe_request_batch": (("req-1", 3), ("req-2", 2)),
            },
        ),
    ):
        _maybe_log_moe_path_hit(layer.layer_name, "gpu_prefill", 5)
        _maybe_log_moe_path_hit(layer.layer_name, "gpu_prefill", 5)

    assert "LvLLM MoE path hit" in caplog_vllm.text
    assert "layer=model.layers.3.mlp.experts" in caplog_vllm.text
    assert "path=gpu_prefill" in caplog_vllm.text
    assert "req-1(3), req-2(2)" in caplog_vllm.text
    assert caplog_vllm.text.count("LvLLM MoE path hit") == 1
