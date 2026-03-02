# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import lru_cache

from .awq_marlin import AwqMarlinFallbackAdapter
from .base import LkQuantAdapter
from .compressed_wna16 import CompressedWna16LkAdapter
from .fp8 import Fp8LkAdapter
from .gguf import GgufLkAdapter
from .unquantized import UnquantizedLkAdapter


@lru_cache(maxsize=1)
def get_lk_quant_adapters() -> tuple[LkQuantAdapter, ...]:
    return (
        UnquantizedLkAdapter(),
        Fp8LkAdapter(),
        CompressedWna16LkAdapter(),
        GgufLkAdapter(),
        AwqMarlinFallbackAdapter(),
    )


def find_lk_quant_adapter(quant_method: object) -> LkQuantAdapter | None:
    for adapter in get_lk_quant_adapters():
        if adapter.matches_quant_method(quant_method):
            return adapter
    return None


def _collect_supported_quant_methods(
    *,
    include_cpu_path: bool | None = None,
    include_gpu_prefill: bool | None = None,
) -> tuple[str, ...]:
    names: set[str] = set()
    for adapter in get_lk_quant_adapters():
        if include_cpu_path is not None and adapter.supports_cpu_path != include_cpu_path:
            continue
        if (
            include_gpu_prefill is not None
            and adapter.supports_gpu_prefill != include_gpu_prefill
        ):
            continue
        names.update(adapter.quant_method_names)
    return tuple(sorted(names))


def get_lk_cpu_supported_quant_method_names() -> tuple[str, ...]:
    return _collect_supported_quant_methods(include_cpu_path=True)


def get_lk_gpu_prefill_supported_quant_method_names() -> tuple[str, ...]:
    return _collect_supported_quant_methods(include_gpu_prefill=True)

