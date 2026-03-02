# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from .base import LkQuantAdapter, LkQuantAdapterSpec


class CompressedWna16LkAdapter(LkQuantAdapter):
    spec = LkQuantAdapterSpec(
        quant_method_names=frozenset(
            {
                "CompressedTensorsWNA16MarlinMoEMethod",
                "CompressedTensorsWNA16MoEMethod",
            }
        ),
        supports_cpu_path=True,
        supports_gpu_prefill=True,
        supports_vllm_fallback=True,
    )

    def to_lk_layout(self, layer) -> None:
        if not hasattr(layer.quant_method, "strategy"):
            raise RuntimeError(
                "Compressed WNA16 quant_method must expose `strategy` "
                "for LK conversion."
            )
        layer._process_compressed_tensors_weights(layer.quant_method.strategy)

    def from_lk_storage(self, layer) -> None:
        param_names = [
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
            "w13_weight_shape",
            "w2_weight_shape",
        ]

        if layer.is_cpu_layer:
            for param_name in param_names:
                setattr(
                    layer,
                    param_name,
                    torch.nn.Parameter(
                        torch.empty(0, device=torch.cuda.current_device()),
                        requires_grad=False,
                    ),
                )
            return

        for param_name in param_names:
            if hasattr(layer, param_name):
                weight = getattr(layer, param_name)
                layer.distribute_weight_tensor(param_name, weight)
            setattr(
                layer,
                param_name,
                torch.nn.Parameter(
                    torch.empty(0, device=torch.cuda.current_device()),
                    requires_grad=False,
                ),
            )
