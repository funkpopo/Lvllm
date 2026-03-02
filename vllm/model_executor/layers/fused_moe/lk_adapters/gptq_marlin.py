# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.logger import init_logger

from .base import LkQuantAdapter, LkQuantAdapterSpec

logger = init_logger(__name__)


class GptqMarlinLkAdapter(LkQuantAdapter):
    spec = LkQuantAdapterSpec(
        quant_method_names=frozenset({"GPTQMarlinMoEMethod"}),
        supports_cpu_path=True,
        supports_gpu_prefill=True,
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
    def _unpack_int32_dim0(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
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
        return unpacked.permute(0, 2, 1).reshape(
            packed.shape[0] * pack_factor, packed.shape[1]
        ).to(torch.float32)

    @staticmethod
    def _resolve_group_indices(
        rows: int,
        groups: int,
        *,
        group_size: int,
        g_idx: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if groups <= 0:
            raise RuntimeError(f"Invalid GPTQ groups={groups}.")
        if groups == 1:
            return torch.zeros(rows, dtype=torch.long, device=device)

        if (
            g_idx is not None
            and g_idx.ndim == 1
            and g_idx.numel() == rows
            and g_idx.numel() > 0
        ):
            idx = g_idx.to(dtype=torch.long, device=device)
        else:
            effective_group_size = rows if group_size == -1 else group_size
            if effective_group_size <= 0:
                raise RuntimeError(f"Invalid GPTQ group_size={group_size}.")
            idx = torch.arange(rows, device=device, dtype=torch.long)
            idx = idx // effective_group_size

        return idx.clamp_(min=0, max=groups - 1)

    @classmethod
    def _dequantize_gptq_matrix(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        *,
        num_bits: int,
        group_size: int,
        is_sym: bool,
    ) -> torch.Tensor:
        quant = cls._unpack_int32_dim0(qweight, num_bits=num_bits)
        rows, cols = quant.shape

        if scales.ndim != 2:
            raise RuntimeError(
                f"Expected 2D scales tensor, but got shape={tuple(scales.shape)}."
            )
        if scales.shape[1] != cols:
            raise RuntimeError(
                "GPTQ scale shape mismatch: "
                f"scales.shape={tuple(scales.shape)} does not match cols={cols}."
            )

        group_indices = cls._resolve_group_indices(
            rows,
            scales.shape[0],
            group_size=group_size,
            g_idx=g_idx,
            device=quant.device,
        )
        scale_rows = scales.to(torch.float32).index_select(0, group_indices)

        if is_sym:
            return quant * scale_rows

        if qzeros is None or qzeros.numel() == 0:
            raise RuntimeError("GPTQ asymmetric dequantization requires qzeros.")

        if torch.is_floating_point(qzeros):
            zeros = qzeros.to(torch.float32)
        else:
            zeros = cls._unpack_int32_last_dim(qzeros, num_bits=num_bits)
        if tuple(zeros.shape) != tuple(scales.shape):
            raise RuntimeError(
                "GPTQ zero-point shape mismatch: "
                f"qzeros.shape={tuple(zeros.shape)} scales.shape={tuple(scales.shape)}."
            )
        zero_rows = zeros.index_select(0, group_indices)
        return (quant - zero_rows) * scale_rows

    @staticmethod
    def _can_use_wna16_fast_path(layer) -> bool:
        quant_config = getattr(layer.quant_method, "quant_config", None)
        if quant_config is None:
            return False
        return (
            getattr(quant_config, "is_sym", False)
            and not getattr(quant_config, "desc_act", False)
            and getattr(quant_config, "group_size", -1) != -1
            and getattr(quant_config, "weight_bits", None) in (4, 8)
        )

    @staticmethod
    def _prepare_wna16_bridge(layer) -> None:
        quant_config = layer.quant_method.quant_config
        layer.w13_weight_packed = torch.nn.Parameter(
            layer.w13_qweight.data, requires_grad=False
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            layer.w2_qweight.data, requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_scales.data, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_scales.data, requires_grad=False
        )

        # LK compressed path expects packed_factor in bits of packed dtype
        # (uint8 -> 8), not values per int32 container.
        layer.quant_method.group_size = quant_config.group_size
        layer.quant_method.num_bits = quant_config.weight_bits
        layer.quant_method.packed_factor = 8

    @classmethod
    def _build_dense_and_process(cls, layer) -> None:
        quant_config = layer.quant_method.quant_config
        weight_bits = getattr(quant_config, "weight_bits", None)
        if weight_bits not in (4, 8):
            raise NotImplementedError(
                f"GPTQ LK adapter does not support weight_bits={weight_bits}."
            )

        output_dtype = layer.moe_config.in_dtype
        num_experts = layer.w13_qweight.shape[0]
        total_intermediate = layer.intermediate_size_per_partition * 2
        hidden_size = layer.hidden_size
        intermediate = layer.intermediate_size_per_partition

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
        w13_g_idx = (
            getattr(layer, "w13_g_idx", None)
            if getattr(quant_config, "desc_act", False)
            else None
        )
        w2_g_idx = (
            getattr(layer, "w2_g_idx", None)
            if getattr(quant_config, "desc_act", False)
            else None
        )

        for expert_idx in range(num_experts):
            w13_dequant = cls._dequantize_gptq_matrix(
                layer.w13_qweight[expert_idx],
                layer.w13_scales[expert_idx],
                None if w13_qzeros is None else w13_qzeros[expert_idx],
                None if w13_g_idx is None else w13_g_idx[expert_idx],
                num_bits=weight_bits,
                group_size=quant_config.group_size,
                is_sym=quant_config.is_sym,
            )
            w13_tensor[expert_idx].copy_(
                w13_dequant.transpose(0, 1).to(dtype=output_dtype, device="cpu")
            )

            w2_dequant = cls._dequantize_gptq_matrix(
                layer.w2_qweight[expert_idx],
                layer.w2_scales[expert_idx],
                None if w2_qzeros is None else w2_qzeros[expert_idx],
                None if w2_g_idx is None else w2_g_idx[expert_idx],
                num_bits=weight_bits,
                group_size=quant_config.group_size,
                is_sym=quant_config.is_sym,
            )
            w2_tensor[expert_idx].copy_(
                w2_dequant.transpose(0, 1).to(dtype=output_dtype, device="cpu")
            )

        layer.w13_weight = torch.nn.Parameter(w13_tensor, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_tensor, requires_grad=False)
        layer._process_regular_weights()

    def to_lk_layout(self, layer) -> None:
        quant_config = getattr(layer.quant_method, "quant_config", None)
        if quant_config is None:
            raise RuntimeError(
                "GPTQMarlinMoEMethod must expose quant_config for LK conversion."
            )
        if not hasattr(layer, "w13_qweight") or not hasattr(layer, "w2_qweight"):
            raise RuntimeError("GPTQ LK conversion requires w13_qweight and w2_qweight.")
        if not hasattr(layer, "w13_scales") or not hasattr(layer, "w2_scales"):
            raise RuntimeError("GPTQ LK conversion requires w13_scales and w2_scales.")

        if self._can_use_wna16_fast_path(layer):
            try:
                from compressed_tensors.quantization import QuantizationStrategy

                self._prepare_wna16_bridge(layer)
                layer._process_compressed_tensors_weights(QuantizationStrategy.GROUP)
                logger.info_once(
                    "Use GPTQ WNA16 LK CPU path for layer=%s quant_method=%s.",
                    layer.layer_name,
                    type(layer.quant_method).__name__,
                )
                return
            except Exception as exc:
                logger.warning_once(
                    "GPTQ WNA16 LK fast path failed on layer=%s (%s). "
                    "Falling back to dense dequantized LK path.",
                    layer.layer_name,
                    exc,
                )

        logger.warning_once(
            "Use GPTQ dense dequantized LK CPU path for layer=%s quant_method=%s. "
            "This improves compatibility but can reduce throughput vs WNA16 path.",
            layer.layer_name,
            type(layer.quant_method).__name__,
        )
        self._build_dense_and_process(layer)

    def from_lk_storage(self, layer) -> None:
        prefill_param_names = (
            "w13_qweight",
            "w2_qweight",
            "w13_scales",
            "w2_scales",
            "w13_qzeros",
            "w2_qzeros",
            "w13_g_idx",
            "w2_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
        )
        cleanup_only_names = (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight",
            "w2_weight",
        )

        device = torch.device(torch.cuda.current_device())
        if layer.is_cpu_layer:
            for name in (*prefill_param_names, *cleanup_only_names):
                if hasattr(layer, name):
                    setattr(
                        layer,
                        name,
                        torch.nn.Parameter(
                            torch.empty(0, device=device), requires_grad=False
                        ),
                    )
            return

        for name in prefill_param_names:
            if not hasattr(layer, name):
                continue
            weight = getattr(layer, name)
            layer.distribute_weight_tensor(name, weight)
            setattr(
                layer,
                name,
                torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False),
            )

        for name in cleanup_only_names:
            if hasattr(layer, name):
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False),
                )
