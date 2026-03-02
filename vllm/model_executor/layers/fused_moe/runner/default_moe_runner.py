# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from vllm.envs import (
    extract_layer_index,
    get_gpu_prefetch_window,
    get_gpu_prefill_min_batch_size,
    is_lk_moe_use_gpu_prefill,
)

logger = init_logger(__name__)


def get_layer_from_name(layer_name: str) -> torch.nn.Module:
    forward_context: ForwardContext = get_forward_context()
    if layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    return forward_context.no_compile_layers[layer_name]


def _is_prefetch_managed_moe_layer(layer_obj: object) -> bool:
    return hasattr(layer_obj, "is_gpu_prefill_layer") and hasattr(
        layer_obj, "is_gpu_resident_layer"
    )


def _fallback_moe_layer_sort_key(layer_name: str) -> tuple[int, int, str]:
    try:
        return (0, extract_layer_index(layer_name), layer_name)
    except Exception:
        # Keep unknown naming patterns deterministic without blocking prefetch.
        return (1, 0, layer_name)


def _get_prefetch_layer_order(
    forward_context: ForwardContext,
) -> tuple[list[str], dict[str, int]]:
    ordered_names = getattr(forward_context, "_moe_prefetch_ordered_layers", None)
    layer_positions = getattr(forward_context, "_moe_prefetch_layer_positions", None)
    if ordered_names is not None and layer_positions is not None:
        return ordered_names, layer_positions

    ordered_names = []
    seen_names: set[str] = set()

    # Prefer static execution order if available.
    source_names = (
        forward_context.all_moe_layers
        if forward_context.all_moe_layers is not None
        else list(forward_context.no_compile_layers.keys())
    )
    for candidate_name in source_names:
        if candidate_name in seen_names:
            continue
        layer_obj = forward_context.no_compile_layers.get(candidate_name)
        if layer_obj is None or not _is_prefetch_managed_moe_layer(layer_obj):
            continue
        ordered_names.append(candidate_name)
        seen_names.add(candidate_name)

    if not ordered_names:
        for candidate_name, layer_obj in forward_context.no_compile_layers.items():
            if candidate_name in seen_names:
                continue
            if not _is_prefetch_managed_moe_layer(layer_obj):
                continue
            ordered_names.append(candidate_name)
            seen_names.add(candidate_name)
        ordered_names.sort(key=_fallback_moe_layer_sort_key)

    layer_positions = {name: idx for idx, name in enumerate(ordered_names)}
    setattr(forward_context, "_moe_prefetch_ordered_layers", ordered_names)
    setattr(forward_context, "_moe_prefetch_layer_positions", layer_positions)
    return ordered_names, layer_positions


def _should_use_lk_cpu_path(layer: torch.nn.Module, hidden_states: torch.Tensor) -> bool:
    if layer.is_gpu_resident_layer:
        return False
    if layer.should_use_gpu_prefill(hidden_states):
        return False

    quant_name = type(layer.quant_method).__name__
    if not layer.supports_lk_cpu_quant_method():
        if layer.supports_vllm_quant_fallback():
            logger.warning_once(
                "Use vLLM quantized MoE path for layer=%s quant_method=%s "
                "because LK CPU path is unsupported.",
                layer.layer_name,
                quant_name,
            )
            return False
        message = (
            "LK MoE CPU execution blocked: unsupported quant_method=%s on layer=%s."
        )
        logger.error(
            message,
            quant_name,
            layer.layer_name,
        )
        raise RuntimeError(message % (quant_name, layer.layer_name))
    if not layer.supports_lk_cpu_path():
        if layer.supports_vllm_quant_fallback():
            logger.warning_once(
                "Use vLLM quantized MoE path for layer=%s quant_method=%s "
                "because LK CPU path is unavailable.",
                layer.layer_name,
                quant_name,
            )
            return False
        message = (
            "LK MoE CPU execution blocked: lk_moe is unavailable for "
            "quant_method=%s on layer=%s."
        )
        logger.error(
            message,
            quant_name,
            layer.layer_name,
        )
        raise RuntimeError(message % (quant_name, layer.layer_name))
    return True


def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    layer = get_layer_from_name(layer_name) 
    layer_name = layer.layer_name
    moe_prefetch(layer, layer_name, hidden_states, forward_context, get_gpu_prefetch_window())
    moe_wait_prefetch(layer, hidden_states, forward_context)
    # TODO(bnell): this can be removed after MK migration is complete.
    layer.ensure_moe_quant_config_init()
    fused_output = layer.runner.forward_impl(
        layer, hidden_states, router_logits, shared_experts_input
    )
    moe_cleanup(layer, layer_name, hidden_states, forward_context)
    return fused_output


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    layer = get_layer_from_name(layer_name) 
    layer_name = layer.layer_name
    moe_prefetch(layer, layer_name, hidden_states, forward_context, get_gpu_prefetch_window())
    moe_wait_prefetch(layer, hidden_states, forward_context)
    # TODO(bnell): this can be removed after MK migration is complete.
    layer.ensure_moe_quant_config_init()
    shared_out, fused_out = layer.runner.forward_impl(
        layer, hidden_states, router_logits, shared_experts_input
    )
    moe_cleanup(layer, layer_name, hidden_states, forward_context)
    return shared_out, fused_out


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Output shapes:
    # - fused_out: same as hidden_states (routed experts use transformed size)
    # - shared_out: same as shared_experts_input if provided, else same as
    #               hidden_states
    # (For latent MoE: shared experts use original hidden_size, not latent size)
    fused_out = torch.empty_like(hidden_states)

    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)

    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class DefaultMoERunner(MoERunner):
    """
    Default implementation of the MoE runner for executing Mixture of Experts layers.

    This class provides a comprehensive implementation for running MoE computations
    with support for:
    - Expert routing and token dispatching
    - Shared experts computation with optional parallel execution using CUDA streams
    - Data parallel (DP) chunking for large batch processing
    - Tensor model parallel and expert parallel operations
    - Various quantization methods and custom operators
    - Both monolithic and decomposed expert execution paths

    The runner handles the complete MoE forward pass including routing tokens to
    experts, executing expert computations, and combining results. It supports
    advanced features like overlapped execution of shared experts and optimized
    kernels for different parallel execution modes.

    Eventually, this class will be split up and specialized for different
    configurations, e.g. the presence or absence of shared experts, a gate, etc.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_input_transform: torch.nn.Module | None,
        gate: torch.nn.Module | None,
        shared_experts: torch.nn.Module | None,
        quant_method: FusedMoEMethodBase,
        reduce_results: bool,
        enable_dbo: bool,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.gate = gate
        self.shared_experts = shared_experts
        self.quant_method = quant_method
        self.reduce_results = reduce_results
        self.enable_dbo = enable_dbo

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            self.shared_experts_stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

        # Needed for string -> FusedMoE layer lookup in custom ops.
        self.layer_name = layer.layer_name

        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped forward_impl.
            if self.shared_experts is None:
                self.moe_forward = _moe_forward
            else:
                self.moe_forward = _moe_forward_shared
        else:
            if self.shared_experts is None:
                self.moe_forward = torch.ops.vllm.moe_forward
            else:
                self.moe_forward = torch.ops.vllm.moe_forward_shared

        # Chunked all2all staging tensor
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_config.moe_parallel_config.use_pplx_kernels
            or self.moe_config.moe_parallel_config.use_deepep_ll_kernels
            or self.moe_config.moe_parallel_config.use_mori_kernels
            or self.moe_config.moe_parallel_config.use_fi_all2allv_kernels
        ) and envs.VLLM_ENABLE_MOE_DP_CHUNK

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        shared_input: torch.Tensor | None,
        has_separate_shared_experts: bool,
        use_chunked_impl: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        use_shared_experts_stream = (
            current_platform.is_cuda()
            and has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        shared_experts_input: torch.Tensor | None = None
        if use_shared_experts_stream:
            assert self.shared_experts_stream is not None
            assert self.moe_config.disable_inplace

            shared_experts_input = (
                shared_input if shared_input is not None else hidden_states
            )

            # Record that the shared_experts_input will be used in the
            # shared_experts_stream to to avoid gc issue from
            # deallocation. For more details:
            # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self.shared_experts_stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, shared_experts_input

    def ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (2, moe.max_num_tokens, self.moe_config.num_logical_experts)
        else:
            states_shape = (moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (moe.max_num_tokens, self.moe_config.num_logical_experts)

        self.batched_hidden_states = torch.zeros(
            states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

        self.batched_router_logits = torch.zeros(
            logits_shape,
            dtype=moe.router_logits_dtype,
            device=torch.cuda.current_device(),
        )

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        assert self.quant_method is not None
        return (
            self.quant_method.moe_mk is not None
            and self.quant_method.moe_mk.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def apply_routed_input_transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by FusedMoE.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        TODO: For latent MoE bandwidth optimization, fc2_latent_proj could be
        moved inside SharedFusedMoE to all-reduce on the smaller latent
        dimension.
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0]
            return result
        return hidden_states

    def _reduce_output(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        trunc_sizes: list[int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        def trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            return x[..., :trunc_size]

        def reduce_and_trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            return trunc(self.maybe_all_reduce_tensor_model_parallel(x), trunc_size)

        if (
            not self.moe_config.is_sequence_parallel
            and not self.use_dp_chunking
            and self.reduce_results
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
        ):
            func = reduce_and_trunc
        else:
            func = trunc

        if isinstance(states, tuple):
            return tuple(
                [func(s, trunc_size) for s, trunc_size in zip(states, trunc_sizes)]
            )
        else:
            assert len(trunc_sizes) == 1
            return func(states, trunc_sizes[0])

    def _encode_layer_name(self) -> str:
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # For latent MoE: save ORIGINAL hidden_states before transform
        # (shared_experts need original dimension, routed experts use transformed)
        if self.shared_experts is not None:
            original_hidden_states = hidden_states
            original_hidden_dim = hidden_states.shape[-1]
        else:
            original_hidden_states = None

        # Apply transform for routed experts (e.g., latent projection for latent MoE)
        hidden_states = self.apply_routed_input_transform(hidden_states)

        # This is the dimension after transform (for routed expert output slicing)
        transformed_hidden_dim = hidden_states.shape[-1]
        if self.moe_config.hidden_dim != transformed_hidden_dim:
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        fused_output = self.moe_forward(
            hidden_states,
            router_logits,
            original_hidden_states,
            self._encode_layer_name(),
        )

        if self.shared_experts is not None:
            orig_hidden_dims = [original_hidden_dim, transformed_hidden_dim]
        else:
            orig_hidden_dims = [transformed_hidden_dim]

        return self._reduce_output(fused_output, orig_hidden_dims)

    def forward_impl_chunked(
        self,
        layer: torch.nn.Module,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        full_shared_input: torch.Tensor | None,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype, (
            f"{self.batched_hidden_states.dtype} == {full_hidden_states.dtype}"
        )
        assert self.batched_router_logits.dtype == full_router_logits.dtype, (
            f"{self.batched_router_logits.dtype} == {full_router_logits.dtype}"
        )
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        # TODO(bnell): Fix shared_expert_inputs w/chunking.
        # assert shared_input is None, (
        #    "Routed input transform is not currently supported with DP chunking."
        # )

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]
            shared_input = (
                full_shared_input[chunk_start:chunk_end, :]
                if full_shared_input is not None
                else None
            )

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            shared_input = (
                shared_input if shared_input is not None else staged_hidden_states
            )

            # Matrix multiply.
            if self.quant_method.is_monolithic:
                assert has_separate_shared_experts or self.shared_experts is None
                if _should_use_lk_cpu_path(layer, hidden_states):
                    topk_weights, topk_ids = self.router.select_experts(
                        hidden_states=staged_hidden_states,
                        router_logits=staged_router_logits,
                    )
                    local_topk_ids = layer.global_to_local_expert_ids(topk_ids) if layer.use_ep else topk_ids
                    final_hidden_states = layer.forward_lk( 
                        staged_hidden_states,
                        topk_weights, 
                        local_topk_ids
                    )
                else:
                    final_hidden_states = self.quant_method.apply_monolithic(
                        layer=layer,
                        x=staged_hidden_states,
                        router_logits=staged_router_logits,
                    )
            else:
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=staged_hidden_states,
                    router_logits=staged_router_logits,
                )
                if _should_use_lk_cpu_path(layer, hidden_states):
                    local_topk_ids = layer.global_to_local_expert_ids(topk_ids) if layer.use_ep else topk_ids
                    final_hidden_states = layer.forward_lk(
                        staged_hidden_states,
                        topk_weights, 
                        local_topk_ids
                    )
                else:
                    final_hidden_states = self.quant_method.apply(
                        layer=layer,
                        x=staged_hidden_states,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        shared_experts_input=shared_input,
                    )

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                shared_output = self.shared_experts(shared_input)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.moe_config.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.moe_config.sp_size
            )

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(
                self.moe_config.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.quant_method is not None

        self.ensure_dp_chunking_init()

        has_separate_shared_experts = (
            not self.quant_method.mk_owns_shared_expert
            and self.shared_experts is not None
        )

        use_chunked_impl = self.use_dp_chunking

        use_shared_experts_stream, shared_experts_input = (
            self._maybe_setup_shared_experts_stream(
                hidden_states,
                shared_input,
                has_separate_shared_experts,
                use_chunked_impl,
            )
        )

        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        if use_chunked_impl:
            return self.forward_impl_chunked(
                layer,
                hidden_states,
                router_logits,
                shared_input,
                has_separate_shared_experts,
            )

        # NOTE(rob): once we finish migrating all the quant methods to use
        # MKs, we can remove the naive dispatch/combine path from here.
        do_naive_dispatch_combine = (
            self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
        )

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            extra_tensors = None
            if do_naive_dispatch_combine:
                post_quant_allgather = (
                    self.quant_method is not None
                    and self.moe_config.dp_size > 1
                    and self.moe_config.use_ep
                    and getattr(self.quant_method, "do_post_quant_allgather", False)
                )
                if post_quant_allgather:
                    hidden_states_to_dispatch, extra_tensors = (
                        self.quant_method.prepare_dp_allgather_tensor(
                            layer, hidden_states, router_logits
                        )
                    )
                else:
                    hidden_states_to_dispatch = hidden_states

                dispatch_res = get_ep_group().dispatch_router_logits(
                    hidden_states_to_dispatch,
                    router_logits,
                    self.moe_config.is_sequence_parallel,
                    extra_tensors=extra_tensors,
                )
                if extra_tensors is not None:
                    (
                        orig_hidden_states,
                        router_logits,
                        extra_tensors_combined,
                    ) = dispatch_res
                    hidden_states_combined = (
                        orig_hidden_states,
                        extra_tensors_combined[0],
                    )
                else:
                    hidden_states_combined, router_logits = dispatch_res
                    orig_hidden_states = hidden_states_combined
            else:
                orig_hidden_states = hidden_states

            # Run shared experts before matrix multiply.
            # because matrix multiply maybe modify the hidden_states.
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                shared_input = (
                    shared_input if shared_input is not None else hidden_states
                )
                shared_output = self.shared_experts(shared_input)

            # NOTE: Similar with DP, PCP also needs dispatch and combine. For
            # simplicity, AgRsAll2All was added separately for PCP here. Maybe
            # we should modify All2AllManager abstract to better support PCP.
            if self.moe_config.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # TODO(bnell): deal with fp4 flashinfer tuple hidden states hack (#30014).
            # Figure out nicer way to do this.
            if do_naive_dispatch_combine:
                x = hidden_states_combined
                x_orig = orig_hidden_states
            else:
                x = hidden_states
                x_orig = hidden_states

            # Matrix multiply.
            if self.quant_method.is_monolithic:
                if _should_use_lk_cpu_path(layer, hidden_states):
                    topk_weights, topk_ids = self.router.select_experts(
                        hidden_states=x,
                        router_logits=router_logits,
                    )
                    local_topk_ids = layer.global_to_local_expert_ids(topk_ids) if layer.use_ep else topk_ids
                    final_hidden_states = layer.forward_lk( 
                        x,
                        topk_weights, 
                        local_topk_ids  
                    )
                else:
                    final_hidden_states = self.quant_method.apply_monolithic(
                        layer=layer,
                        x=x,
                        router_logits=router_logits,
                    )
            else:
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=x_orig,
                    router_logits=router_logits,
                )
                if _should_use_lk_cpu_path(layer, hidden_states):
                    local_topk_ids = layer.global_to_local_expert_ids(topk_ids) if layer.use_ep else topk_ids
                    final_hidden_states = layer.forward_lk(
                        x,
                        topk_weights, 
                        local_topk_ids
                    )
                else:
                    final_hidden_states = self.quant_method.apply(
                        layer=layer,
                        x=x,  # The type signture of this is wrong due to the hack.
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        shared_experts_input=shared_input,
                    )

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(shared_experts_input)
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(
                        states, self.moe_config.is_sequence_parallel
                    )

                if self.moe_config.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            if self.shared_experts is not None:
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            else:
                return combine_output(final_hidden_states)
    
    
def moe_cleanup(layer, layer_name: str, hidden_states: torch.Tensor, 
                forward_context: ForwardContext): 
    if torch.cuda.is_current_stream_capturing():
        return
    
    if not is_lk_moe_use_gpu_prefill():
        return
    
    if hidden_states.size(0) < get_gpu_prefill_min_batch_size():
        return
    
    if not layer.should_use_gpu_prefill(hidden_states):
        return
    
    _, layer_positions = _get_prefetch_layer_order(forward_context)
    layer_position = layer_positions.get(layer_name)
    if layer_position is None:
        return

    batch_key = id(forward_context.batch_descriptor)
     
    if not hasattr(forward_context, '_batch_prefetch_states'):
        return
    if batch_key not in forward_context._batch_prefetch_states:
        return
    
    batch_state = forward_context._batch_prefetch_states[batch_key]
    prefetched_layers = batch_state['prefetched_layers']
     
    keys_to_clean = [
        candidate_name
        for candidate_name in list(prefetched_layers)
        if layer_positions.get(candidate_name, -1) <= layer_position
    ]
    
    for candidate_name in keys_to_clean:
        layer_obj = forward_context.no_compile_layers.get(candidate_name)
        if layer_obj and layer_obj.is_gpu_resident_layer:
            prefetched_layers.discard(candidate_name)
            continue
        if layer_obj:
            moe_clean_gpu_prefill(layer_obj)
        prefetched_layers.discard(candidate_name)
        if hasattr(forward_context, '_prefetch_events'):  
            if layer_obj:
                layer_id = id(layer_obj)
                if layer_id in forward_context._prefetch_events:
                    del forward_context._prefetch_events[layer_id]


def _disable_layer_gpu_prefill(layer: torch.nn.Module, reason: str) -> None:
    if getattr(layer, "is_gpu_prefill_layer", False):
        layer.is_gpu_prefill_layer = False
    layer_name = getattr(layer, "layer_name", "<unknown>")
    quant_method = getattr(layer, "quant_method", None)
    quant_name = type(quant_method).__name__ if quant_method is not None else "<none>"
    logger.warning_once(
        "Disable LK GPU prefill for layer=%s quant_method=%s. Reason: %s",
        layer_name,
        quant_name,
        reason,
    )


def moe_prefetch(layer, layer_name: str, hidden_states: torch.Tensor, 
                 forward_context: ForwardContext, gpu_prefetch_window: int): 
    if torch.cuda.is_current_stream_capturing():
        return
    
    if not is_lk_moe_use_gpu_prefill():
        return
    
    if hidden_states.size(0) < get_gpu_prefill_min_batch_size():
        return
    
    if not layer.should_use_gpu_prefill(hidden_states):
        return

    supports_gpu_prefill = getattr(layer, "supports_lk_gpu_prefill", None)
    if callable(supports_gpu_prefill) and not supports_gpu_prefill():
        _disable_layer_gpu_prefill(
            layer,
            "quant_method is unsupported by LK GPU prefill or lk_moe is unavailable",
        )
        return
    
    if not hasattr(forward_context, '_prefetch_stream'):
        forward_context._prefetch_stream = torch.cuda.Stream()
        
    if not hasattr(forward_context, '_prefetch_events'):
        forward_context._prefetch_events = {}  # layer_id -> event
    
    ordered_names, layer_positions = _get_prefetch_layer_order(forward_context)
    layer_position = layer_positions.get(layer_name)
    if layer_position is None:
        return

    batch_key = id(forward_context.batch_descriptor) 
    
    if not hasattr(forward_context, '_batch_prefetch_states'):
        forward_context._batch_prefetch_states = {}
    
    if batch_key not in forward_context._batch_prefetch_states:
        forward_context._batch_prefetch_states[batch_key] = {
            'prefetched_layers': set(),
            'called_layers': set(),
            'last_position': -1,
        }
    
    batch_state = forward_context._batch_prefetch_states[batch_key]
    prefetched_layers = batch_state['prefetched_layers']
    called_layers = batch_state['called_layers']
    last_position = batch_state['last_position']
     
    if layer_position <= last_position:
        prefetched_layers.clear()
        called_layers.clear()
    batch_state['last_position'] = layer_position
     
    if layer_name in called_layers:
        return
    
    called_layers.add(layer_name)  
            
    active_prefetches = 0
    for candidate_name in prefetched_layers:
        candidate_layer = forward_context.no_compile_layers.get(candidate_name)
        if candidate_layer is not None and not candidate_layer.is_gpu_resident_layer:
            active_prefetches += 1
     
    available_slots = gpu_prefetch_window - active_prefetches
    
    if available_slots > 0: 
        prefetch_candidates = []
        for candidate_name in ordered_names[layer_position:]:
            if len(prefetch_candidates) >= available_slots:
                break
            if candidate_name in prefetched_layers:
                continue
            candidate_layer = forward_context.no_compile_layers.get(candidate_name)
            if candidate_layer is None:
                continue
            if candidate_layer.is_gpu_resident_layer:
                continue
            candidate_supports_gpu_prefill = getattr(
                candidate_layer, "supports_lk_gpu_prefill", None
            )
            if (
                callable(candidate_supports_gpu_prefill)
                and not candidate_supports_gpu_prefill()
            ):
                _disable_layer_gpu_prefill(
                    candidate_layer,
                    "quant_method is unsupported by LK GPU prefill or lk_moe is unavailable",
                )
                continue
            if candidate_layer.is_gpu_prefill_layer:
                prefetch_candidates.append((candidate_name, candidate_layer))
         
        for candidate_name, layer_obj in prefetch_candidates:
            moe_prepare_gpu_prefill(layer_obj, forward_context, torch.cuda.current_device())
            prefetched_layers.add(candidate_name)
            
def collect_weight_from_moe(layer, param_name: str) -> torch.Tensor:
    pin_memory = is_pin_memory_available()
    shape_name = param_name + "_origin_shape"
    dtype_name = param_name + "_origin_dtype"
    if hasattr(layer, shape_name) and hasattr(layer, dtype_name):
        origin_shape = getattr(layer, shape_name)
        origin_dtype = getattr(layer, dtype_name)

        # Reuse pinned host staging buffers and shape metadata across prefetch
        # cycles to reduce allocator pressure in long-context workloads.
        if not hasattr(layer, "_lk_prefetch_cpu_staging"):
            layer._lk_prefetch_cpu_staging = {}
        if not hasattr(layer, "_lk_prefetch_shape_cache"):
            layer._lk_prefetch_shape_cache = {}
        if not hasattr(layer, "_lk_prefetch_shape_tuple_cache"):
            layer._lk_prefetch_shape_tuple_cache = {}

        origin_shape_tuple = tuple(origin_shape)

        weight_cpu = layer._lk_prefetch_cpu_staging.get(param_name)
        if (
            weight_cpu is None
            or tuple(weight_cpu.shape) != origin_shape_tuple
            or weight_cpu.dtype != origin_dtype
        ):
            weight_cpu = torch.empty(
                origin_shape,
                dtype=origin_dtype,
                device="cpu",
                requires_grad=False,
                pin_memory=pin_memory,
            ).contiguous()
            layer._lk_prefetch_cpu_staging[param_name] = weight_cpu

        shape_array = layer._lk_prefetch_shape_cache.get(param_name)
        cached_shape = layer._lk_prefetch_shape_tuple_cache.get(param_name)
        if shape_array is None or cached_shape != origin_shape_tuple:
            shape_array = torch.tensor(origin_shape, dtype=torch.int64).contiguous()
            layer._lk_prefetch_shape_cache[param_name] = shape_array
            layer._lk_prefetch_shape_tuple_cache[param_name] = origin_shape_tuple

        layer.lk_moe.collectWeight(
            param_name,
            weight_cpu.data_ptr(),
            shape_array.data_ptr(),
            weight_cpu[0].nbytes,
        )
        return weight_cpu

    raise RuntimeError(
        "Missing LK weight metadata for param=%s on layer=%s."
        % (param_name, layer.layer_name)
    )


def _copy_or_replace_param(
    owner: object,
    param_name: str,
    weight_cpu: torch.Tensor,
    device: torch.device,
) -> None:
    existing = getattr(owner, param_name, None)
    if (
        isinstance(existing, torch.nn.Parameter)
        and tuple(existing.shape) == tuple(weight_cpu.shape)
        and existing.dtype == weight_cpu.dtype
        and existing.device == device
    ):
        existing.data.copy_(weight_cpu, non_blocking=True)
        return

    setattr(
        owner,
        param_name,
        torch.nn.Parameter(
            weight_cpu.to(device, non_blocking=True),
            requires_grad=False,
        ),
    )


def _empty_param(owner: object, param_name: str, device: torch.device) -> None:
    if not hasattr(owner, param_name):
        return
    setattr(
        owner,
        param_name,
        torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False),
    )


def moe_prepare_gpu_prefill_fp8(
    layer, forward_context: ForwardContext, device: torch.device
):
    del forward_context
    param_names = [
        "w13_weight",
        "w2_weight",
    ]

    block_quant = getattr(layer.quant_method, "block_quant", False)
    scale_names = [
        "w13_weight_scale_inv" if block_quant else "w13_weight_scale",
        "w2_weight_scale_inv" if block_quant else "w2_weight_scale",
    ]

    quant_config_names = [
        "w1_scale",
        "w2_scale",
    ]

    for param_name in param_names:
        weight_cpu = collect_weight_from_moe(layer, param_name)
        _copy_or_replace_param(layer, param_name, weight_cpu, device)

    use_quant_config = (
        hasattr(layer, "moe_quant_config")
        and hasattr(layer.moe_quant_config, quant_config_names[0])
        and hasattr(layer.moe_quant_config, quant_config_names[1])
    )

    if use_quant_config:
        for scale_name in quant_config_names:
            weight_cpu = collect_weight_from_moe(layer, scale_name)
            _copy_or_replace_param(layer.moe_quant_config, scale_name, weight_cpu, device)
    else:
        for scale_name in scale_names:
            weight_cpu = collect_weight_from_moe(layer, scale_name)
            _copy_or_replace_param(layer, scale_name, weight_cpu, device)


def moe_clean_gpu_prefill_fp8(layer):
    param_names = [
        "w13_weight",
        "w2_weight",
    ]

    block_quant = getattr(layer.quant_method, "block_quant", False)
    scale_names = [
        "w13_weight_scale_inv" if block_quant else "w13_weight_scale",
        "w2_weight_scale_inv" if block_quant else "w2_weight_scale",
    ]

    quant_config_names = [
        "w1_scale",
        "w2_scale",
    ]
    device = torch.device(torch.cuda.current_device())

    for param_name in param_names:
        _empty_param(layer, param_name, device)

    use_quant_config = (
        hasattr(layer, "moe_quant_config")
        and hasattr(layer.moe_quant_config, quant_config_names[0])
        and hasattr(layer.moe_quant_config, quant_config_names[1])
    )

    if use_quant_config:
        for scale_name in quant_config_names:
            _empty_param(layer.moe_quant_config, scale_name, device)
    else:
        for scale_name in scale_names:
            _empty_param(layer, scale_name, device)


def moe_prepare_gpu_prefill_channel_scale(
    layer, forward_context: ForwardContext, device: torch.device
):
    del forward_context
    for param_name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
    ):
        weight_cpu = collect_weight_from_moe(layer, param_name)
        _copy_or_replace_param(layer, param_name, weight_cpu, device)


def moe_clean_gpu_prefill_channel_scale(layer):
    device = torch.device(torch.cuda.current_device())
    for param_name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
    ):
        _empty_param(layer, param_name, device)


def moe_prepare_gpu_prefill_wna16(
    layer, forward_context: ForwardContext, device: torch.device
):
    del forward_context
    param_names = [
        "w13_weight_packed",
        "w2_weight_packed",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_g_idx_sort_indices",
        "w2_g_idx_sort_indices",
        "w13_weight_shape",
        "w2_weight_shape",
    ]

    for param_name in param_names:
        weight_cpu = collect_weight_from_moe(layer, param_name)
        _copy_or_replace_param(layer, param_name, weight_cpu, device)


def moe_clean_gpu_prefill_wna16(layer):
    param_names = [
        "w13_weight_packed",
        "w2_weight_packed",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_g_idx_sort_indices",
        "w2_g_idx_sort_indices",
        "w13_weight_shape",
        "w2_weight_shape",
    ]
    device = torch.device(torch.cuda.current_device())
    for param_name in param_names:
        _empty_param(layer, param_name, device)


def moe_prepare_gpu_prefill_regular(
    layer, forward_context: ForwardContext, device: torch.device
):
    del forward_context
    pin_memory = is_pin_memory_available()

    w13_shape = (
        layer.global_num_experts,
        layer.intermediate_size_per_partition * 2,
        layer.hidden_size,
    )
    w2_shape = (
        layer.global_num_experts,
        layer.hidden_size,
        layer.intermediate_size_per_partition,
    )
    weight_dtype = layer.moe_config.in_dtype

    w13_weight_cpu = getattr(layer, "_lk_prefetch_regular_w13", None)
    if (
        w13_weight_cpu is None
        or tuple(w13_weight_cpu.shape) != w13_shape
        or w13_weight_cpu.dtype != weight_dtype
    ):
        w13_weight_cpu = torch.zeros(
            w13_shape,
            dtype=weight_dtype,
            device="cpu",
            requires_grad=False,
            pin_memory=pin_memory,
        ).contiguous()
        layer._lk_prefetch_regular_w13 = w13_weight_cpu

    w2_weight_cpu = getattr(layer, "_lk_prefetch_regular_w2", None)
    if (
        w2_weight_cpu is None
        or tuple(w2_weight_cpu.shape) != w2_shape
        or w2_weight_cpu.dtype != weight_dtype
    ):
        w2_weight_cpu = torch.zeros(
            w2_shape,
            dtype=weight_dtype,
            device="cpu",
            requires_grad=False,
            pin_memory=pin_memory,
        ).contiguous()
        layer._lk_prefetch_regular_w2 = w2_weight_cpu

    layer.lk_moe.collect_weights(
        True,
        0,
        0,
        w13_weight_cpu.data_ptr(),
        0,
    )
    layer.lk_moe.collect_weights(
        True,
        0,
        0,
        w13_weight_cpu.data_ptr(),
        1,
    )
    layer.lk_moe.collect_weights(
        True,
        0,
        0,
        w2_weight_cpu.data_ptr(),
        2,
    )

    _copy_or_replace_param(layer, "w13_weight", w13_weight_cpu, device)
    _copy_or_replace_param(layer, "w2_weight", w2_weight_cpu, device)


def moe_clean_gpu_prefill_regular(layer):
    device = torch.device(torch.cuda.current_device())
    _empty_param(layer, "w13_weight", device)
    _empty_param(layer, "w2_weight", device)


_GPU_PREFILL_PREPARE_HANDLERS: dict[
    str, Callable[[torch.nn.Module, ForwardContext, torch.device], None]
] = {
    "UnquantizedFusedMoEMethod": moe_prepare_gpu_prefill_regular,
    "CompressedTensorsWNA16MarlinMoEMethod": moe_prepare_gpu_prefill_wna16,
    "CompressedTensorsWNA16MoEMethod": moe_prepare_gpu_prefill_wna16,
    "Fp8MoEMethod": moe_prepare_gpu_prefill_fp8,
    "CompressedTensorsW8A8Fp8MoEMethod": moe_prepare_gpu_prefill_fp8,
    "CompressedTensorsW8A8Int8MoEMethod": moe_prepare_gpu_prefill_channel_scale,
    "ExpertsInt8MoEMethod": moe_prepare_gpu_prefill_channel_scale,
}

_GPU_PREFILL_CLEAN_HANDLERS: dict[str, Callable[[torch.nn.Module], None]] = {
    "UnquantizedFusedMoEMethod": moe_clean_gpu_prefill_regular,
    "CompressedTensorsWNA16MarlinMoEMethod": moe_clean_gpu_prefill_wna16,
    "CompressedTensorsWNA16MoEMethod": moe_clean_gpu_prefill_wna16,
    "Fp8MoEMethod": moe_clean_gpu_prefill_fp8,
    "CompressedTensorsW8A8Fp8MoEMethod": moe_clean_gpu_prefill_fp8,
    "CompressedTensorsW8A8Int8MoEMethod": moe_clean_gpu_prefill_channel_scale,
    "ExpertsInt8MoEMethod": moe_clean_gpu_prefill_channel_scale,
}


def _resolve_gpu_prefill_handler(layer, action: str):
    quant_method_name = type(layer.quant_method).__name__
    if not layer.supports_lk_gpu_prefill_quant_method():
        raise ValueError(layer.format_lk_gpu_prefill_unsupported_message())

    handlers = (
        _GPU_PREFILL_PREPARE_HANDLERS
        if action == "prepare"
        else _GPU_PREFILL_CLEAN_HANDLERS
    )
    handler = handlers.get(quant_method_name)
    if handler is not None:
        return handler

    support_matrix = layer.get_lk_gpu_prefill_support_matrix()
    raise RuntimeError(
        "Missing LK GPU prefill %s handler for quant_method=%s on layer=%s. "
        "Support matrix=%s"
        % (action, quant_method_name, layer.layer_name, support_matrix)
    )


def moe_prepare_gpu_prefill(layer, forward_context: ForwardContext, device: torch.device):
    if layer.is_gpu_prefill_layer:
        with torch.no_grad():
            prefetch_stream = forward_context._prefetch_stream
            prefetch_events = forward_context._prefetch_events

            with torch.cuda.stream(prefetch_stream):
                try:
                    prepare_handler = _resolve_gpu_prefill_handler(
                        layer, action="prepare"
                    )
                except Exception as e:
                    _disable_layer_gpu_prefill(layer, str(e))
                    return
                prepare_handler(layer, forward_context, device)

                layer_id = id(layer)
                event = torch.cuda.Event()
                event.record(prefetch_stream)
                prefetch_events[layer_id] = event


def moe_clean_gpu_prefill(layer):
    with torch.no_grad():
        try:
            clean_handler = _resolve_gpu_prefill_handler(layer, action="clean")
        except Exception as e:
            _disable_layer_gpu_prefill(layer, str(e))
            return
        clean_handler(layer)

def moe_wait_prefetch(layer, hidden_states: torch.Tensor, forward_context: ForwardContext):
    if torch.cuda.is_current_stream_capturing():
        return 
    if not hasattr(forward_context, '_prefetch_events'):
        return 
    if not layer.should_use_gpu_prefill(hidden_states):
        return
    layer_id = id(layer)
    prefetch_events = forward_context._prefetch_events
    if layer_id in prefetch_events:
        prefetch_events[layer_id].wait()
        del prefetch_events[layer_id] 
    
