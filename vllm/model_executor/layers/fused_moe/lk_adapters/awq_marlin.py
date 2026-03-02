# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from .base import LkQuantAdapter, LkQuantAdapterSpec


class AwqMarlinFallbackAdapter(LkQuantAdapter):
    spec = LkQuantAdapterSpec(
        quant_method_names=frozenset({"AWQMarlinMoEMethod"}),
        supports_cpu_path=False,
        supports_gpu_prefill=False,
        supports_vllm_fallback=True,
    )

    def to_lk_layout(self, layer) -> None:
        raise NotImplementedError(
            "AWQ weights are not supported by LK MoE conversion yet. "
            "Please disable LK CPU mode for this layer or use a supported quantization."
        )

    def from_lk_storage(self, layer) -> None:
        return
