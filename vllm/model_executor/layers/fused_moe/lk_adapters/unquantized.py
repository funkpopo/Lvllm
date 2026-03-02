# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from .base import LkQuantAdapter, LkQuantAdapterSpec


class UnquantizedLkAdapter(LkQuantAdapter):
    spec = LkQuantAdapterSpec(
        quant_method_names=frozenset({"UnquantizedFusedMoEMethod"}),
        supports_cpu_path=True,
        supports_gpu_prefill=True,
        supports_vllm_fallback=True,
    )

    def to_lk_layout(self, layer) -> None:
        layer._process_regular_weights()

    def from_lk_storage(self, layer) -> None:
        if hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight"):
            layer.w13_weight = torch.nn.Parameter(
                torch.empty(0, device=torch.cuda.current_device()),
                requires_grad=False,
            )
            layer.w2_weight = torch.nn.Parameter(
                torch.empty(0, device=torch.cuda.current_device()),
                requires_grad=False,
            )
