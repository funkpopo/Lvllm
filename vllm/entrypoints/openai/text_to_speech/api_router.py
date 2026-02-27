# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.text_to_speech.protocol import SpeechRequest
from vllm.entrypoints.openai.text_to_speech.serving import OpenAIServingTextToSpeech
from vllm.entrypoints.utils import load_aware_call, with_cancellation

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger

router = APIRouter()


def text_to_speech(request: Request) -> OpenAIServingTextToSpeech | None:
    return getattr(request.app.state, "openai_serving_text_to_speech", None)


@router.post(
    "/v1/audio/speech",
    responses={
        HTTPStatus.OK.value: {"content": {"audio/wav": {}, "audio/pcm": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech(request: SpeechRequest, raw_request: Request):
    handler = text_to_speech(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Speech API"
        )

    result = await handler.create_speech(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    audio_bytes, media_type = result
    return Response(content=audio_bytes, media_type=media_type)


@router.get(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def list_voices(raw_request: Request):
    handler = text_to_speech(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support voices API"
        )

    result = await handler.get_voices()
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)


def init_speech_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
):
    state.openai_serving_text_to_speech = OpenAIServingTextToSpeech(
        engine_client,
        state.openai_serving_models,
        request_logger=request_logger,
        log_error_stack=args.log_error_stack,
        tts_max_instructions_length=args.tts_max_instructions_length,
    )
