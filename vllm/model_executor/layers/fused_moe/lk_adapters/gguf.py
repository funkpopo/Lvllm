# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from .base import LkQuantAdapter, LkQuantAdapterSpec


class GgufLkAdapter(LkQuantAdapter):
    spec = LkQuantAdapterSpec(
        quant_method_names=frozenset({"GGUFMoEMethod"}),
        supports_cpu_path=True,
        supports_gpu_prefill=False,
        supports_vllm_fallback=True,
    )

    def to_lk_layout(self, layer) -> None:
        layer._process_gguf_weights()

    def from_lk_storage(self, layer) -> None:
        # GGUF conversion consumes qweight tensors directly; no extra
        # post-processing is required after LK runtime init.
        return
