# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.envs import MoeComputeStrategy, get_moe_compute_strategy

from .base import LkQuantAdapter, LkQuantAdapterSpec


class Fp8LkAdapter(LkQuantAdapter):
    spec = LkQuantAdapterSpec(
        quant_method_names=frozenset(
            {
                "Fp8MoEMethod",
                "CompressedTensorsW8A8Fp8MoEMethod",
            }
        ),
        supports_cpu_path=True,
        supports_gpu_prefill=True,
        supports_vllm_fallback=True,
    )

    def to_lk_layout(self, layer) -> None:
        quant_method_name = type(layer.quant_method).__name__
        strategy = get_moe_compute_strategy()
        if quant_method_name == "Fp8MoEMethod":
            if strategy == MoeComputeStrategy.KEEP:
                layer._process_fp8_weights(layer.quant_method.block_quant)
            elif strategy == MoeComputeStrategy.TO_DTYPE:
                layer._process_block_weights()
            else:
                layer._process_block_weights_quant(strategy)
            return

        if strategy == MoeComputeStrategy.KEEP:
            layer._process_fp8_weights(False)
        elif strategy == MoeComputeStrategy.TO_DTYPE:
            layer._process_channel_weights()
        else:
            layer._process_channel_weights_quant(strategy)

    def from_lk_storage(self, layer) -> None:
        param_names = [
            "w13_weight",
            "w2_weight",
        ]
        scale_names = [
            "w13_weight_scale_inv"
            if layer.quant_method.block_quant
            else "w13_weight_scale",
            "w2_weight_scale_inv" if layer.quant_method.block_quant else "w2_weight_scale",
        ]
        quant_config_names = [
            "w1_scale",
            "w2_scale",
        ]

        if layer.is_cpu_layer:
            for param_name in param_names:
                if hasattr(layer, param_name):
                    setattr(
                        layer,
                        param_name,
                        torch.nn.Parameter(
                            torch.empty(0, device=torch.cuda.current_device()),
                            requires_grad=False,
                        ),
                    )
            for scale_name in scale_names:
                if hasattr(layer, scale_name):
                    setattr(
                        layer,
                        scale_name,
                        torch.nn.Parameter(
                            torch.empty(0, device=torch.cuda.current_device()),
                            requires_grad=False,
                        ),
                    )
            for quant_config_name in quant_config_names:
                if hasattr(layer, "moe_quant_config") and hasattr(
                    layer.moe_quant_config, quant_config_name
                ):
                    setattr(
                        layer.moe_quant_config,
                        quant_config_name,
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

        has_quant_config = (
            hasattr(layer, "moe_quant_config")
            and hasattr(layer.moe_quant_config, quant_config_names[0])
            and hasattr(layer.moe_quant_config, quant_config_names[1])
        )
        if has_quant_config:
            for scale_name in quant_config_names:
                if hasattr(layer.moe_quant_config, scale_name):
                    weight = getattr(layer.moe_quant_config, scale_name)
                    layer.distribute_weight_tensor(scale_name, weight)
                    setattr(
                        layer.moe_quant_config,
                        scale_name,
                        torch.nn.Parameter(
                            torch.empty(0, device=torch.cuda.current_device()),
                            requires_grad=False,
                        ),
                    )

            for scale_name in scale_names:
                if hasattr(layer, scale_name):
                    setattr(
                        layer,
                        scale_name,
                        torch.nn.Parameter(
                            torch.empty(0, device=torch.cuda.current_device()),
                            requires_grad=False,
                        ),
                    )
            return

        for scale_name in scale_names:
            if hasattr(layer, scale_name):
                weight = getattr(layer, scale_name)
                layer.distribute_weight_tensor(scale_name, weight)
                setattr(
                    layer,
                    scale_name,
                    torch.nn.Parameter(
                        torch.empty(0, device=torch.cuda.current_device()),
                        requires_grad=False,
                    ),
                )
