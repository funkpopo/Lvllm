# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from collections.abc import AsyncGenerator
from io import BytesIO
from typing import Any, cast

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

    @staticmethod
    def _encode_with_ffmpeg(
        pcm_bytes: bytes,
        sr: int,
        fmt: str,
    ) -> tuple[bytes, str]:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                f"response_format='{fmt}' requires ffmpeg in PATH."
            )

        out_format = fmt
        codec_args: list[str] = []
        media_type = "application/octet-stream"
        if fmt == "mp3":
            out_format = "mp3"
            media_type = "audio/mpeg"
        elif fmt == "aac":
            out_format = "adts"
            codec_args = ["-c:a", "aac", "-b:a", "128k"]
            media_type = "audio/aac"
        elif fmt == "opus":
            out_format = "ogg"
            codec_args = ["-c:a", "libopus", "-b:a", "96k", "-vbr", "on"]
            media_type = "audio/opus"
        else:
            raise ValueError(f"Unsupported ffmpeg output format: {fmt}")

        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            *codec_args,
            "-f",
            out_format,
            "pipe:1",
        ]
        proc = subprocess.run(
            cmd,
            input=pcm_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"ffmpeg failed for format '{fmt}': {stderr.strip() or 'unknown error'}"
            )
        return proc.stdout, media_type

    def _serialize_audio(self, wav: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
        if fmt == "pcm":
            return self._to_pcm16le_bytes(wav), "audio/pcm"

        if fmt in {"mp3", "aac", "opus"}:
            pcm_bytes = self._to_pcm16le_bytes(wav)
            return self._encode_with_ffmpeg(pcm_bytes, sr, fmt)

        if isinstance(sf, PlaceholderModule):
            raise RuntimeError(
                "Audio encoding requires `soundfile`. "
                "Install with: pip install soundfile"
            )

        if fmt not in {"wav", "flac"}:
            raise ValueError(
                f"response_format='{fmt}' is not supported yet. "
                "Use one of: wav, flac, pcm, mp3, aac, opus."
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

    async def create_speech_stream(
        self,
        request: SpeechRequest,
        raw_request: Request | None = None,
    ) -> tuple[AsyncGenerator[bytes, None], str] | ErrorResponse:
        if not request.stream:
            return self.create_error_response("stream must be true for streaming API.")
        if request.response_format != "pcm":
            return self.create_error_response(
                "Streaming currently supports PCM only.", param="response_format"
            )

        if not self._is_qwen3_tts_arch():
            return self.create_error_response(
                "The loaded model does not support Text-to-Speech API.",
            )

        if error_check_ret := await self._check_model(request):
            return error_check_ret

        if not request.input.strip():
            return self.create_error_response(
                "The 'input' field must be non-empty.", param="input"
            )

        try:
            tts_model = self._load_qwen_tts_model()
            core_model = getattr(tts_model, "model", None)
            stream_pcm = getattr(core_model, "stream_generate_pcm", None)
            if not callable(stream_pcm):
                return self.create_error_response(
                    "This qwen-tts build does not expose true streaming "
                    "(missing model.stream_generate_pcm). "
                    "Please upgrade qwen-tts or use a streaming-enabled build."
                )

            task_type = cast(
                Qwen3TTSTaskType, request.task_type or self._infer_default_task_type()
            )
            language = request.language or "Auto"
            instructions = (request.instructions or "")[: self.tts_max_instructions_length]

            # Build inputs in the same shape expected by Qwen3-TTS stream API.
            build_assistant = getattr(tts_model, "_build_assistant_text", None)
            tokenize_texts = getattr(tts_model, "_tokenize_texts", None)
            build_instruct = getattr(tts_model, "_build_instruct_text", None)
            merge_kwargs = getattr(tts_model, "_merge_generate_kwargs", None)
            if not all(callable(x) for x in [build_assistant, tokenize_texts, merge_kwargs]):
                return self.create_error_response(
                    "Current qwen-tts wrapper is missing helper methods required "
                    "for streaming."
                )

            input_text = build_assistant(request.input)
            input_ids = tokenize_texts([input_text])
            languages = [language]

            generation_kwargs: dict[str, Any] = {}
            if request.max_new_tokens is not None:
                generation_kwargs["max_new_tokens"] = request.max_new_tokens
            generation_kwargs = merge_kwargs(**generation_kwargs)
            # Streaming API accepts a narrower subset.
            allowed = {
                "do_sample",
                "top_k",
                "top_p",
                "temperature",
                "subtalker_dosample",
                "subtalker_top_k",
                "subtalker_top_p",
                "subtalker_temperature",
            }
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if k in allowed}

            stream_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "languages": languages,
                "non_streaming_mode": False,
                **generation_kwargs,
            }
            if request.emit_every_frames is not None:
                stream_kwargs["emit_every_frames"] = request.emit_every_frames
            if request.decode_window_frames is not None:
                stream_kwargs["decode_window_frames"] = request.decode_window_frames
            if request.overlap_samples is not None:
                stream_kwargs["overlap_samples"] = request.overlap_samples
            if request.max_frames is not None:
                stream_kwargs["max_frames"] = request.max_frames

            if task_type == "CustomVoice":
                stream_kwargs["speakers"] = [request.voice]
                if instructions:
                    if not callable(build_instruct):
                        return self.create_error_response(
                            "Current qwen-tts wrapper does not support instruction "
                            "tokenization for streaming."
                        )
                    stream_kwargs["instruct_ids"] = tokenize_texts(
                        [build_instruct(instructions)]
                    )
            elif task_type == "VoiceDesign":
                if instructions:
                    if not callable(build_instruct):
                        return self.create_error_response(
                            "Current qwen-tts wrapper does not support instruction "
                            "tokenization for streaming."
                        )
                    stream_kwargs["instruct_ids"] = tokenize_texts(
                        [build_instruct(instructions)]
                    )
            else:
                create_vc_prompt = getattr(tts_model, "create_voice_clone_prompt", None)
                to_vc_dict = getattr(tts_model, "_prompt_items_to_voice_clone_prompt", None)
                build_ref_text = getattr(tts_model, "_build_ref_text", None)
                if not (
                    callable(create_vc_prompt)
                    and callable(to_vc_dict)
                    and callable(build_ref_text)
                ):
                    return self.create_error_response(
                        "Current qwen-tts wrapper does not support Base streaming helpers."
                    )
                if not request.ref_audio:
                    return self.create_error_response(
                        "`ref_audio` is required for Base streaming mode.",
                        param="ref_audio",
                    )

                prompt_items = create_vc_prompt(
                    ref_audio=request.ref_audio,
                    ref_text=request.ref_text,
                    x_vector_only_mode=bool(request.x_vector_only_mode),
                )
                voice_clone_prompt = to_vc_dict(prompt_items)
                stream_kwargs["voice_clone_prompt"] = voice_clone_prompt
                if request.ref_text:
                    stream_kwargs["ref_ids"] = tokenize_texts([build_ref_text(request.ref_text)])

            async def _stream() -> AsyncGenerator[bytes, None]:
                for chunk, _sr in stream_pcm(**stream_kwargs):
                    yield self._to_pcm16le_bytes(np.asarray(chunk))

            return _stream(), "audio/pcm"
        except Exception as e:
            logger.exception("TTS streaming generation failed.")
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
