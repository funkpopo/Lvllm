# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.envs import MoeComputeStrategy, get_moe_compute_strategy
from vllm.logger import init_logger

from .base import LkQuantAdapter, LkQuantAdapterSpec

logger = init_logger(__name__)


class Int8W8A8LkAdapter(LkQuantAdapter):
    spec = LkQuantAdapterSpec(
        quant_method_names=frozenset(
            {
                "CompressedTensorsW8A8Int8MoEMethod",
                "ExpertsInt8MoEMethod",
            }
        ),
        supports_cpu_path=True,
        supports_gpu_prefill=True,
        supports_vllm_fallback=True,
    )

    @staticmethod
    def _reshape_channel_scale(
        scale: torch.Tensor,
        *,
        expected_experts: int,
        expected_channels: int,
        scale_name: str,
    ) -> torch.Tensor:
        if scale.ndim == 3:
            expected_shape = (expected_experts, expected_channels, 1)
            if tuple(scale.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected {scale_name} shape={tuple(scale.shape)}; "
                    f"expected {expected_shape}."
                )
            return scale.contiguous()

        if scale.ndim == 2:
            expected_shape = (expected_experts, expected_channels)
            if tuple(scale.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected {scale_name} shape={tuple(scale.shape)}; "
                    f"expected {expected_shape}."
                )
            return scale.unsqueeze(-1).contiguous()

        if scale.ndim == 1:
            expected_shape = (expected_experts,)
            if tuple(scale.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected {scale_name} shape={tuple(scale.shape)}; "
                    f"expected {expected_shape}."
                )
            return (
                scale.view(expected_experts, 1, 1)
                .expand(expected_experts, expected_channels, 1)
                .contiguous()
            )

        raise RuntimeError(
            f"Unsupported {scale_name} ndim={scale.ndim}; "
            "expected 1D/2D/3D channel scales."
        )

    @staticmethod
    def _normalize_channel_scales(layer) -> None:
        if hasattr(layer, "w13_weight_scale") and hasattr(layer, "w2_weight_scale"):
            w13_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale
        elif hasattr(layer, "w13_scale") and hasattr(layer, "w2_scale"):
            w13_scale = layer.w13_scale
            w2_scale = layer.w2_scale
        else:
            raise RuntimeError(
                "INT8 W8A8 LK conversion requires either "
                "(w13_weight_scale, w2_weight_scale) or (w13_scale, w2_scale)."
            )

        num_experts = layer.w13_weight.shape[0]
        w13_channels = layer.w13_weight.shape[1]
        w2_channels = layer.w2_weight.shape[1]

        w13_scale = Int8W8A8LkAdapter._reshape_channel_scale(
            w13_scale,
            expected_experts=num_experts,
            expected_channels=w13_channels,
            scale_name="w13_scale",
        )
        w2_scale = Int8W8A8LkAdapter._reshape_channel_scale(
            w2_scale,
            expected_experts=num_experts,
            expected_channels=w2_channels,
            scale_name="w2_scale",
        )

        setattr(
            layer,
            "w13_weight_scale",
            torch.nn.Parameter(w13_scale, requires_grad=False),
        )
        setattr(
            layer,
            "w2_weight_scale",
            torch.nn.Parameter(w2_scale, requires_grad=False),
        )

    def to_lk_layout(self, layer) -> None:
        self._normalize_channel_scales(layer)

        strategy = get_moe_compute_strategy()
        if strategy == MoeComputeStrategy.KEEP:
            logger.warning_once(
                "LVLLM_MOE_USE_WEIGHT=KEEP is unsupported for INT8 W8A8 LK MoE. "
                "Falling back to TO_DTYPE.",
            )
            strategy = MoeComputeStrategy.TO_DTYPE

        if strategy == MoeComputeStrategy.TO_DTYPE:
            layer._process_channel_weights()
        else:
            layer._process_channel_weights_quant(strategy)

    def from_lk_storage(self, layer) -> None:
        to_distribute = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        )
        aliases = ("w13_scale", "w2_scale")

        if layer.is_cpu_layer:
            for param_name in (*to_distribute, *aliases):
                if hasattr(layer, param_name):
                    setattr(
                        layer,
                        param_name,
                        torch.nn.Parameter(
                            torch.empty(0, device=torch.cuda.current_device()),
                            requires_grad=False,
                        ),
                    )
            return

        for param_name in to_distribute:
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

        for param_name in aliases:
            if hasattr(layer, param_name):
                setattr(
                    layer,
                    param_name,
                    torch.nn.Parameter(
                        torch.empty(0, device=torch.cuda.current_device()),
                        requires_grad=False,
                    ),
                )
