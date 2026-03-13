# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import pytest
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from vllm.v1.metrics.lvllm import (
    configure_lvllm_metrics,
    observe_cpu_moe_tokens,
    observe_fallback_count,
    observe_gpu_prefetch_active_layers,
    observe_gpu_prefetch_bytes,
    observe_gpu_prefetch_wait_seconds,
    observe_gpu_resident_moe_tokens,
    observe_lk_moe_forward_latency_seconds,
    reset_lvllm_metrics_for_test,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics


@pytest.fixture(autouse=True)
def clean_lvllm_metrics():
    unregister_vllm_metrics()
    reset_lvllm_metrics_for_test()
    yield
    unregister_vllm_metrics()
    reset_lvllm_metrics_for_test()


def _find_sample(
    metric_text: str,
    family_name: str,
    sample_name: str,
) -> tuple[float, dict[str, str]]:
    for family in text_string_to_metric_families(metric_text):
        if family.name != family_name:
            continue
        for sample in family.samples:
            if sample.name == sample_name:
                return sample.value, sample.labels
    raise AssertionError(f"Missing sample {sample_name!r} in {family_name!r}")


def _find_all_samples(
    metric_text: str,
    family_name: str,
    sample_name: str,
) -> Iterable[tuple[float, dict[str, str]]]:
    for family in text_string_to_metric_families(metric_text):
        if family.name != family_name:
            continue
        for sample in family.samples:
            if sample.name == sample_name:
                yield sample.value, sample.labels


def test_lvllm_metrics_registered_and_observed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LVLLM_MOE_NUMA_ENABLED", "1")
    monkeypatch.setenv("LVLLM_ENABLE_NUMA_INTERLEAVE", "1")
    monkeypatch.setenv("LVLLM_GPU_RESIDENT_MOE_LAYERS", "1,3-5")
    monkeypatch.setenv("LVLLM_GPU_PREFILL_MIN_BATCH_SIZE", "128")
    monkeypatch.setenv("LVLLM_GPU_PREFETCH_WINDOW", "4")

    configure_lvllm_metrics("test-model")

    observe_lk_moe_forward_latency_seconds(0.25)
    observe_gpu_prefetch_bytes(4096)
    observe_gpu_prefetch_wait_seconds(0.01)
    observe_gpu_prefetch_active_layers(3)
    observe_cpu_moe_tokens(128)
    observe_gpu_resident_moe_tokens(64)
    observe_fallback_count()

    metric_text = generate_latest().decode("utf-8")

    value, labels = _find_sample(
        metric_text,
        "vllm:lvllm_cpu_moe_tokens",
        "vllm:lvllm_cpu_moe_tokens_total",
    )
    assert value == 128
    assert labels["model_name"] == "test-model"

    value, _ = _find_sample(
        metric_text,
        "vllm:lvllm_gpu_resident_moe_tokens",
        "vllm:lvllm_gpu_resident_moe_tokens_total",
    )
    assert value == 64

    value, _ = _find_sample(
        metric_text,
        "vllm:lvllm_gpu_prefetch_bytes",
        "vllm:lvllm_gpu_prefetch_bytes_total",
    )
    assert value == 4096

    value, _ = _find_sample(
        metric_text,
        "vllm:lvllm_gpu_prefetch_active_layers",
        "vllm:lvllm_gpu_prefetch_active_layers",
    )
    assert value == 3

    value, _ = _find_sample(
        metric_text,
        "vllm:lvllm_fallback_count",
        "vllm:lvllm_fallback_count_total",
    )
    assert value == 1

    histogram_samples = list(
        _find_all_samples(
            metric_text,
            "vllm:lvllm_lk_moe_forward_latency_seconds",
            "vllm:lvllm_lk_moe_forward_latency_seconds_count",
        )
    )
    assert histogram_samples
    assert histogram_samples[0][0] == 1

    histogram_samples = list(
        _find_all_samples(
            metric_text,
            "vllm:lvllm_gpu_prefetch_wait_seconds",
            "vllm:lvllm_gpu_prefetch_wait_seconds_count",
        )
    )
    assert histogram_samples
    assert histogram_samples[0][0] == 1

    value, labels = _find_sample(
        metric_text,
        "vllm:lvllm_numa_node_plan",
        "vllm:lvllm_numa_node_plan",
    )
    assert value == 1
    assert labels["model_name"] == "test-model"
    assert labels["numa_enabled"] == "1"
    assert labels["numa_policy"] == "interleave"
    assert labels["gpu_resident_moe_layers"] == "1,3-5"
    assert labels["gpu_prefill_min_batch_size"] == "128"
    assert labels["gpu_prefetch_window"] == "4"
