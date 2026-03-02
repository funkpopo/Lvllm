# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.logger import init_logger

from .base import LkQuantAdapter, LkQuantAdapterSpec

logger = init_logger(__name__)


class AwqMarlinLkAdapter(LkQuantAdapter):
    spec = LkQuantAdapterSpec(
        quant_method_names=frozenset({"AWQMarlinMoEMethod"}),
        supports_cpu_path=True,
        supports_gpu_prefill=False,
        supports_vllm_fallback=True,
    )

    @staticmethod
    def _unpack_int32_last_dim(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
        if packed.ndim != 2:
            raise RuntimeError(
                f"Expected 2D packed tensor, but got shape={tuple(packed.shape)}."
            )
        pack_factor = 32 // num_bits
        shifts = (
            torch.arange(pack_factor, device=packed.device, dtype=torch.int32) * num_bits
        )
        unpacked = torch.bitwise_and(
            torch.bitwise_right_shift(packed.to(torch.int32).unsqueeze(-1), shifts),
            (1 << num_bits) - 1,
        )
        return unpacked.reshape(packed.shape[0], packed.shape[1] * pack_factor).to(
            torch.float32
        )

    @staticmethod
    def _dequantize_awq_matrix(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor | None,
        *,
        group_size: int,
        zero_point: bool,
    ) -> torch.Tensor:
        quant = AwqMarlinLkAdapter._unpack_int32_last_dim(qweight, num_bits=4)
        rows, cols = quant.shape

        if scales.ndim != 2:
            raise RuntimeError(
                f"Expected 2D scales tensor, but got shape={tuple(scales.shape)}."
            )
        if scales.shape[1] != cols:
            raise RuntimeError(
                "AWQ scale shape mismatch: "
                f"scales.shape={tuple(scales.shape)} does not match cols={cols}."
            )

        if group_size == -1:
            group_size = rows
        if group_size <= 0:
            raise RuntimeError(f"Invalid AWQ group_size={group_size}.")

        num_groups = scales.shape[0]
        if num_groups == 1:
            group_indices = torch.zeros(rows, dtype=torch.long, device=quant.device)
        else:
            group_indices = torch.arange(rows, device=quant.device, dtype=torch.long)
            group_indices = group_indices // group_size
            if int(group_indices[-1].item()) >= num_groups:
                raise RuntimeError(
                    "AWQ group index overflow: "
                    f"rows={rows}, group_size={group_size}, num_groups={num_groups}."
                )

        scale_rows = scales.to(torch.float32).index_select(0, group_indices)
        if not zero_point or qzeros is None or qzeros.numel() == 0:
            return quant * scale_rows

        if qzeros.ndim != 2:
            raise RuntimeError(
                f"Expected 2D qzeros tensor, but got shape={tuple(qzeros.shape)}."
            )
        zeros = AwqMarlinLkAdapter._unpack_int32_last_dim(qzeros, num_bits=4)
        if tuple(zeros.shape) != tuple(scales.shape):
            raise RuntimeError(
                "AWQ zero-point shape mismatch: "
                f"qzeros.shape={tuple(zeros.shape)} scales.shape={tuple(scales.shape)}."
            )
        zero_rows = zeros.index_select(0, group_indices)
        return (quant - zero_rows) * scale_rows

    def to_lk_layout(self, layer) -> None:
        quant_config = getattr(layer.quant_method, "quant_config", None)
        if quant_config is None:
            raise RuntimeError(
                "AWQMarlinMoEMethod must expose quant_config for LK conversion."
            )
        if getattr(quant_config, "weight_bits", None) != 4:
            raise NotImplementedError(
                "AWQ LK adapter currently supports only 4-bit weights."
            )
        if not hasattr(layer, "w13_qweight") or not hasattr(layer, "w2_qweight"):
            raise RuntimeError("AWQ LK conversion requires w13_qweight and w2_qweight.")
        if not hasattr(layer, "w13_scales") or not hasattr(layer, "w2_scales"):
            raise RuntimeError("AWQ LK conversion requires w13_scales and w2_scales.")

        output_dtype = layer.moe_config.in_dtype
        num_experts = layer.w13_qweight.shape[0]
        total_intermediate = layer.intermediate_size_per_partition * 2
        hidden_size = layer.hidden_size
        intermediate = layer.intermediate_size_per_partition
        group_size = getattr(quant_config, "group_size", -1)
        use_zero_point = bool(getattr(quant_config, "zero_point", True))

        w13_tensor = torch.empty(
            (num_experts, total_intermediate, hidden_size),
            dtype=output_dtype,
            device="cpu",
            requires_grad=False,
        )
        w2_tensor = torch.empty(
            (num_experts, hidden_size, intermediate),
            dtype=output_dtype,
            device="cpu",
            requires_grad=False,
        )

        w13_qzeros = getattr(layer, "w13_qzeros", None)
        w2_qzeros = getattr(layer, "w2_qzeros", None)

        for expert_idx in range(num_experts):
            w13_dequant = self._dequantize_awq_matrix(
                layer.w13_qweight[expert_idx],
                layer.w13_scales[expert_idx],
                None if w13_qzeros is None else w13_qzeros[expert_idx],
                group_size=group_size,
                zero_point=use_zero_point,
            )
            w13_tensor[expert_idx].copy_(
                w13_dequant.transpose(0, 1).to(dtype=output_dtype, device="cpu")
            )

            w2_dequant = self._dequantize_awq_matrix(
                layer.w2_qweight[expert_idx],
                layer.w2_scales[expert_idx],
                None if w2_qzeros is None else w2_qzeros[expert_idx],
                group_size=group_size,
                zero_point=use_zero_point,
            )
            w2_tensor[expert_idx].copy_(
                w2_dequant.transpose(0, 1).to(dtype=output_dtype, device="cpu")
            )

        layer.w13_weight = torch.nn.Parameter(w13_tensor, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_tensor, requires_grad=False)

        logger.info_once(
            "Use AWQ dequantized LK CPU path for layer=%s quant_method=%s. "
            "GPU prefill remains disabled for this quantization.",
            layer.layer_name,
            type(layer.quant_method).__name__,
        )
        layer._process_regular_weights()

    def from_lk_storage(self, layer) -> None:
        param_names = (
            "w13_qweight",
            "w2_qweight",
            "w13_scales",
            "w2_scales",
            "w13_qzeros",
            "w2_qzeros",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
            "w13_weight",
            "w2_weight",
        )

        device = torch.device(torch.cuda.current_device())
        for name in param_names:
            if not hasattr(layer, name):
                continue
            setattr(
                layer,
                name,
                torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False),
            )
