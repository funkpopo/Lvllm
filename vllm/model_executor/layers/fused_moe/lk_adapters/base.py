# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE


@dataclass(frozen=True)
class LkQuantAdapterSpec:
    quant_method_names: frozenset[str]
    supports_cpu_path: bool = True
    supports_gpu_prefill: bool = False
    supports_vllm_fallback: bool = True


class LkQuantAdapter(ABC):
    spec: LkQuantAdapterSpec

    @property
    def quant_method_names(self) -> frozenset[str]:
        return self.spec.quant_method_names

    @property
    def supports_cpu_path(self) -> bool:
        return self.spec.supports_cpu_path

    @property
    def supports_gpu_prefill(self) -> bool:
        return self.spec.supports_gpu_prefill

    @property
    def supports_vllm_fallback(self) -> bool:
        return self.spec.supports_vllm_fallback

    def matches_quant_method(self, quant_method: object) -> bool:
        return type(quant_method).__name__ in self.quant_method_names

    @abstractmethod
    def to_lk_layout(self, layer: "FusedMoE") -> None:
        """Convert current quantized weights into LK runtime layout."""

    @abstractmethod
    def from_lk_storage(self, layer: "FusedMoE") -> None:
        """Release or offload tensors after LK layout has been materialized."""
