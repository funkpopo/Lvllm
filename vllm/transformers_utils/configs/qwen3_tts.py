# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class Qwen3TTSTalkerCodePredictorConfig(PretrainedConfig):
    model_type = "qwen3_tts_talker_code_predictor"

    def __init__(
        self,
        vocab_size=2048,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=5,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=65536,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(**kwargs)


class Qwen3TTSTalkerConfig(PretrainedConfig):
    model_type = "qwen3_tts_talker"
    sub_configs = {
        "code_predictor_config": Qwen3TTSTalkerCodePredictorConfig,
    }

    def __init__(
        self,
        vocab_size=3072,
        text_vocab_size=151936,
        hidden_size=1024,
        text_hidden_size=2048,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_code_groups=16,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=True,
        position_id_per_seconds=13,
        code_predictor_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.hidden_size = hidden_size
        self.text_hidden_size = text_hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_code_groups = num_code_groups
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.position_id_per_seconds = position_id_per_seconds

        if isinstance(code_predictor_config, dict):
            code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(
                **code_predictor_config
            )
        elif code_predictor_config is None:
            code_predictor_config = Qwen3TTSTalkerCodePredictorConfig()
        self.code_predictor_config = code_predictor_config

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(**kwargs)


class Qwen3TTSConfig(PretrainedConfig):
    model_type = "qwen3_tts"
    sub_configs = {
        "talker_config": Qwen3TTSTalkerConfig,
    }

    def __init__(
        self,
        talker_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if talker_config is None:
            talker_config = {}
        self.talker_config = Qwen3TTSTalkerConfig(**talker_config)

    def get_text_config(self, decoder=False) -> PretrainedConfig:  # noqa: ARG002
        return self.talker_config


__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
]
