from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context as forward_context_module
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer


def _run_should_use_gpu_prefill(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_gpu_prefill_layer: bool,
    batch_size: int,
    min_batch_size: int,
    is_stream_capturing: bool,
    cudagraph_mode: CUDAGraphMode | None,
) -> bool:
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: is_stream_capturing,
    )
    monkeypatch.setattr(
        fused_moe_layer,
        "get_gpu_prefill_min_batch_size",
        lambda: min_batch_size,
    )

    if cudagraph_mode is None:
        monkeypatch.setattr(
            forward_context_module,
            "is_forward_context_available",
            lambda: False,
        )

        def _should_not_be_called():
            raise AssertionError("get_forward_context should not be called")

        monkeypatch.setattr(
            forward_context_module, "get_forward_context", _should_not_be_called
        )
    else:
        monkeypatch.setattr(
            forward_context_module,
            "is_forward_context_available",
            lambda: True,
        )
        monkeypatch.setattr(
            forward_context_module,
            "get_forward_context",
            lambda: SimpleNamespace(cudagraph_runtime_mode=cudagraph_mode),
        )

    dummy_layer = SimpleNamespace(is_gpu_prefill_layer=is_gpu_prefill_layer)
    hidden_states = torch.zeros((batch_size, 8), dtype=torch.float32)
    return FusedMoE.should_use_gpu_prefill(dummy_layer, hidden_states)


def test_should_use_gpu_prefill_disabled_in_cudagraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        _run_should_use_gpu_prefill(
            monkeypatch,
            is_gpu_prefill_layer=True,
            batch_size=32,
            min_batch_size=16,
            is_stream_capturing=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )
        is False
    )


@pytest.mark.parametrize(
    "batch_size,min_batch_size,expected",
    [
        (7, 8, False),
        (8, 8, True),
        (9, 8, True),
    ],
)
def test_should_use_gpu_prefill_threshold_boundary(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    min_batch_size: int,
    expected: bool,
) -> None:
    assert (
        _run_should_use_gpu_prefill(
            monkeypatch,
            is_gpu_prefill_layer=True,
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            is_stream_capturing=False,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        is expected
    )


def test_should_use_gpu_prefill_disabled_for_non_prefill_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        _run_should_use_gpu_prefill(
            monkeypatch,
            is_gpu_prefill_layer=False,
            batch_size=64,
            min_batch_size=1,
            is_stream_capturing=False,
            cudagraph_mode=None,
        )
        is False
    )


def test_should_use_gpu_prefill_disabled_while_stream_capturing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        _run_should_use_gpu_prefill(
            monkeypatch,
            is_gpu_prefill_layer=True,
            batch_size=64,
            min_batch_size=1,
            is_stream_capturing=True,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        is False
    )
