# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import platform
import threading
from dataclasses import dataclass
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

import vllm.envs as envs

_LVLLM_METRICS_LOCK = threading.Lock()
_LVLLM_METRICS: "_LvllmPrometheusMetrics | None" = None

_LATENCY_BUCKETS_SECONDS = [
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
]


@dataclass
class _LvllmPrometheusMetrics:
    model_name: str
    lk_moe_forward_latency_seconds: Any
    gpu_prefetch_bytes: Any
    gpu_prefetch_wait_seconds: Any
    gpu_prefetch_active_layers: Any
    cpu_moe_tokens: Any
    gpu_resident_moe_tokens: Any
    fallback_count: Any


def _get_allowed_numa_nodes_label() -> str:
    if not envs.LVLLM_MOE_NUMA_ENABLED:
        return "disabled"

    if platform.system() != "Linux":
        return "unknown"

    try:
        from vllm.platforms.cpu import CpuPlatform

        allowed_nodes, _ = CpuPlatform.get_allowed_cpu_core_node_list()
        if not allowed_nodes:
            return "unknown"
        return ",".join(str(node) for node in allowed_nodes)
    except Exception:
        return "unknown"


def _get_numa_policy_label() -> str:
    if not envs.LVLLM_MOE_NUMA_ENABLED:
        return "disabled"
    if envs.LVLLM_ENABLE_NUMA_INTERLEAVE:
        return "interleave"
    return "membind"


def _get_gpu_resident_layer_plan_label() -> str:
    plan = os.getenv("LVLLM_GPU_RESIDENT_MOE_LAYERS", "").strip()
    return plan or "none"


def configure_lvllm_metrics(
    model_name: str,
    *,
    gauge_cls: type[Gauge] = Gauge,
    counter_cls: type[Counter] = Counter,
    histogram_cls: type[Histogram] = Histogram,
) -> None:
    global _LVLLM_METRICS

    with _LVLLM_METRICS_LOCK:
        if _LVLLM_METRICS is not None:
            return

        labelnames = ["model_name"]

        lk_moe_forward_latency_seconds = histogram_cls(
            name="vllm:lvllm_lk_moe_forward_latency_seconds",
            documentation="Histogram of LvLLM lk_moe forward latency.",
            buckets=_LATENCY_BUCKETS_SECONDS,
            labelnames=labelnames,
        ).labels(model_name=model_name)

        gpu_prefetch_bytes = counter_cls(
            name="vllm:lvllm_gpu_prefetch_bytes",
            documentation="Total bytes moved to GPU by LvLLM MoE prefetch.",
            labelnames=labelnames,
        ).labels(model_name=model_name)

        gpu_prefetch_wait_seconds = histogram_cls(
            name="vllm:lvllm_gpu_prefetch_wait_seconds",
            documentation=(
                "Histogram of GPU-side wait inserted before consuming "
                "LvLLM prefetched MoE weights."
            ),
            buckets=_LATENCY_BUCKETS_SECONDS,
            labelnames=labelnames,
        ).labels(model_name=model_name)

        gpu_prefetch_active_layers = gauge_cls(
            name="vllm:lvllm_gpu_prefetch_active_layers",
            documentation="Number of active LvLLM GPU-prefetched MoE layers.",
            multiprocess_mode="sum",
            labelnames=labelnames,
        ).labels(model_name=model_name)

        cpu_moe_tokens = counter_cls(
            name="vllm:lvllm_cpu_moe_tokens",
            documentation="Total tokens routed through the LvLLM CPU MoE path.",
            labelnames=labelnames,
        ).labels(model_name=model_name)

        gpu_resident_moe_tokens = counter_cls(
            name="vllm:lvllm_gpu_resident_moe_tokens",
            documentation=(
                "Total tokens routed through the LvLLM GPU-resident MoE path."
            ),
            labelnames=labelnames,
        ).labels(model_name=model_name)

        fallback_count = counter_cls(
            name="vllm:lvllm_fallback_count",
            documentation="Number of LvLLM lk_moe forward fallback events.",
            labelnames=labelnames,
        ).labels(model_name=model_name)

        numa_node_plan = gauge_cls(
            name="vllm:lvllm_numa_node_plan",
            documentation=(
                "Static LvLLM NUMA and MoE placement plan exported as labels."
            ),
            multiprocess_mode="mostrecent",
            labelnames=[
                "model_name",
                "numa_enabled",
                "numa_policy",
                "allowed_numa_nodes",
                "gpu_resident_moe_layers",
                "gpu_prefill_min_batch_size",
                "gpu_prefetch_window",
            ],
        )

        gpu_prefetch_active_layers.set(0)
        numa_node_plan.labels(
            model_name=model_name,
            numa_enabled=str(int(bool(envs.LVLLM_MOE_NUMA_ENABLED))),
            numa_policy=_get_numa_policy_label(),
            allowed_numa_nodes=_get_allowed_numa_nodes_label(),
            gpu_resident_moe_layers=_get_gpu_resident_layer_plan_label(),
            gpu_prefill_min_batch_size=str(envs.LVLLM_GPU_PREFILL_MIN_BATCH_SIZE),
            gpu_prefetch_window=str(envs.LVLLM_GPU_PREFETCH_WINDOW),
        ).set(1)

        _LVLLM_METRICS = _LvllmPrometheusMetrics(
            model_name=model_name,
            lk_moe_forward_latency_seconds=lk_moe_forward_latency_seconds,
            gpu_prefetch_bytes=gpu_prefetch_bytes,
            gpu_prefetch_wait_seconds=gpu_prefetch_wait_seconds,
            gpu_prefetch_active_layers=gpu_prefetch_active_layers,
            cpu_moe_tokens=cpu_moe_tokens,
            gpu_resident_moe_tokens=gpu_resident_moe_tokens,
            fallback_count=fallback_count,
        )


def reset_lvllm_metrics_for_test() -> None:
    global _LVLLM_METRICS
    with _LVLLM_METRICS_LOCK:
        _LVLLM_METRICS = None


def _get_metrics() -> _LvllmPrometheusMetrics | None:
    return _LVLLM_METRICS


def observe_lk_moe_forward_latency_seconds(duration_seconds: float) -> None:
    metrics = _get_metrics()
    if metrics is None:
        return
    metrics.lk_moe_forward_latency_seconds.observe(duration_seconds)


def observe_gpu_prefetch_bytes(num_bytes: int) -> None:
    metrics = _get_metrics()
    if metrics is None or num_bytes <= 0:
        return
    metrics.gpu_prefetch_bytes.inc(num_bytes)


def observe_gpu_prefetch_wait_seconds(duration_seconds: float) -> None:
    metrics = _get_metrics()
    if metrics is None or duration_seconds < 0:
        return
    metrics.gpu_prefetch_wait_seconds.observe(duration_seconds)


def observe_gpu_prefetch_active_layers(active_layers: int) -> None:
    metrics = _get_metrics()
    if metrics is None:
        return
    metrics.gpu_prefetch_active_layers.set(max(active_layers, 0))


def observe_cpu_moe_tokens(num_tokens: int) -> None:
    metrics = _get_metrics()
    if metrics is None or num_tokens <= 0:
        return
    metrics.cpu_moe_tokens.inc(num_tokens)


def observe_gpu_resident_moe_tokens(num_tokens: int) -> None:
    metrics = _get_metrics()
    if metrics is None or num_tokens <= 0:
        return
    metrics.gpu_resident_moe_tokens.inc(num_tokens)


def observe_fallback_count(increment: int = 1) -> None:
    metrics = _get_metrics()
    if metrics is None or increment <= 0:
        return
    metrics.fallback_count.inc(increment)
