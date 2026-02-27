# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
from vllm.exceptions import VLLMValidationError

AudioSpeechResponseFormat: TypeAlias = Literal["wav", "flac", "pcm", "mp3", "opus", "aac"]
Qwen3TTSTaskType: TypeAlias = Literal["CustomVoice", "VoiceDesign", "Base"]


class SpeechRequest(OpenAIBaseModel):
    # OpenAI-compatible fields.
    model: str | None = None
    input: str
    voice: str = "vivian"
    response_format: AudioSpeechResponseFormat = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    # Qwen3-TTS extensions.
    task_type: Qwen3TTSTaskType | None = None
    language: str | None = None
    instructions: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)
    emit_every_frames: int | None = Field(default=None, ge=1)
    decode_window_frames: int | None = Field(default=None, ge=1)
    overlap_samples: int | None = Field(default=None, ge=0)
    max_frames: int | None = Field(default=None, ge=1)
    stream: bool = False

    @model_validator(mode="after")
    def _validate_stream_constraints(self) -> "SpeechRequest":
        if self.stream:
            if self.response_format != "pcm":
                raise VLLMValidationError(
                    "stream=true requires response_format='pcm'.",
                    parameter="response_format",
                    value=self.response_format,
                )
            if self.speed != 1.0:
                raise VLLMValidationError(
                    "speed is not supported when stream=true.",
                    parameter="speed",
                    value=self.speed,
                )

        return self


class VoicesResponse(OpenAIBaseModel):
    voices: list[str]
