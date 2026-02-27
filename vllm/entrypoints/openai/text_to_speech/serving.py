# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from typing import Any, Literal, cast

import numpy as np
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.text_to_speech.protocol import (
    Qwen3TTSTaskType,
    SpeechRequest,
    VoicesResponse,
)
from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

try:
    import soundfile as sf
except ImportError:
    sf = PlaceholderModule("soundfile")  # type: ignore[assignment]

logger = init_logger(__name__)


class OpenAIServingTextToSpeech(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        log_error_stack: bool = False,
        tts_max_instructions_length: int = 500,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )
        self.tts_max_instructions_length = tts_max_instructions_length
        self._qwen_tts_model: Any | None = None

    def _is_qwen3_tts_arch(self) -> bool:
        archs = getattr(self.model_config.hf_config, "architectures", None) or []
        return "Qwen3TTSForConditionalGeneration" in archs

    def _infer_default_task_type(self) -> Qwen3TTSTaskType:
        model_name = str(self.model_config.model).lower()
        if model_name.endswith("-voicedesign"):
            return "VoiceDesign"
        if model_name.endswith("-base"):
            return "Base"
        return "CustomVoice"

    def _load_qwen_tts_model(self) -> Any:
        if self._qwen_tts_model is not None:
            return self._qwen_tts_model

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise RuntimeError(
                "Qwen3-TTS serving requires `qwen-tts` and `torch`. "
                "Install with: pip install -U qwen-tts"
            ) from e

        attn_impl = "flash_attention_2"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn_impl = None

        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

        self._qwen_tts_model = Qwen3TTSModel.from_pretrained(
            self.model_config.model, **kwargs
        )
        return self._qwen_tts_model

    @staticmethod
    def _to_pcm16le_bytes(wav: np.ndarray) -> bytes:
        audio = np.asarray(wav, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767.0).astype(np.int16)
        return audio_i16.tobytes()

    def _serialize_audio(self, wav: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
        if fmt == "pcm":
            return self._to_pcm16le_bytes(wav), "audio/pcm"

        if isinstance(sf, PlaceholderModule):
            raise RuntimeError(
                "Audio encoding requires `soundfile`. "
                "Install with: pip install soundfile"
            )

        # `soundfile` does not reliably support all OpenAI formats in every env.
        # Keep deterministic behaviour for formats we can encode directly.
        if fmt not in {"wav", "flac"}:
            raise ValueError(
                f"response_format='{fmt}' is not supported yet. "
                "Use one of: wav, flac, pcm."
            )

        with BytesIO() as buf:
            sf.write(buf, wav, sr, format=fmt.upper())
            return buf.getvalue(), f"audio/{fmt}"

    async def create_speech(
        self,
        request: SpeechRequest,
        raw_request: Request | None = None,
    ) -> tuple[bytes, str] | ErrorResponse:
        if not self._is_qwen3_tts_arch():
            return self.create_error_response(
                "The loaded model does not support Text-to-Speech API.",
            )

        if error_check_ret := await self._check_model(request):
            return error_check_ret

        if request.stream:
            return self.create_error_response(
                "stream=true is not implemented in this backend yet."
            )

        if not request.input.strip():
            return self.create_error_response(
                "The 'input' field must be non-empty.", param="input"
            )

        try:
            tts_model = self._load_qwen_tts_model()
            task_type = cast(
                Qwen3TTSTaskType, request.task_type or self._infer_default_task_type()
            )
            language = request.language or "Auto"
            instructions = (request.instructions or "")[: self.tts_max_instructions_length]

            generation_kwargs: dict[str, Any] = {}
            if request.max_new_tokens is not None:
                generation_kwargs["max_new_tokens"] = request.max_new_tokens

            if task_type == "VoiceDesign":
                wavs, sr = tts_model.generate_voice_design(
                    request.input,
                    language=language,
                    instruct=instructions,
                    **generation_kwargs,
                )
            elif task_type == "Base":
                if request.ref_audio:
                    generation_kwargs["ref_audio"] = request.ref_audio
                if request.ref_text:
                    generation_kwargs["ref_text"] = request.ref_text
                if request.x_vector_only_mode is not None:
                    generation_kwargs["x_vector_only_mode"] = request.x_vector_only_mode

                wavs, sr = tts_model.generate_voice_clone(
                    request.input,
                    language=language,
                    **generation_kwargs,
                )
            else:
                wavs, sr = tts_model.generate_custom_voice(
                    request.input,
                    speaker=request.voice,
                    language=language,
                    instruct=instructions,
                    **generation_kwargs,
                )

            if not wavs:
                return self.create_error_response("TTS model did not produce audio output.")

            audio_bytes, media_type = self._serialize_audio(
                np.asarray(wavs[0]), int(sr), request.response_format
            )
            return audio_bytes, media_type
        except Exception as e:
            logger.exception("TTS generation failed.")
            return self.create_error_response(e)

    async def get_voices(self) -> VoicesResponse | ErrorResponse:
        if not self._is_qwen3_tts_arch():
            return self.create_error_response(
                "The loaded model does not support voice listing."
            )

        try:
            tts_model = self._load_qwen_tts_model()
            speakers = []
            if hasattr(tts_model, "get_supported_speakers"):
                speakers = tts_model.get_supported_speakers() or []
            return VoicesResponse(voices=[str(s) for s in speakers])
        except Exception as e:
            logger.exception("Voice listing failed.")
            return self.create_error_response(e)
