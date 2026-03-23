import asyncio
from pathlib import Path

import pytest

from app.types import ModelState
from unittest.mock import AsyncMock
from typing import AsyncGenerator

import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from app.server import app as real_app
from app.state import AppState
from unittest.mock import patch, MagicMock

BASE_URL = "http://localhost:8000"

@pytest.fixture
def mock_db_models():
    models_rows = [
        {"id_model": 1, "model_name": "tiny"},
        {"id_model": 2, "model_name": "base"},
        {"id_model": 3, "model_name": "small"},
        {"id_model": 4, "model_name": "medium"},
    ]
    enabled_rows = [
        {"model_id": 1},
        {"model_id": 3},
    ]

    with patch("app.model_registry.fetch_all", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = [models_rows, enabled_rows]

        with patch("app.model_registry.ModelRegistry._download_model", new_callable=AsyncMock) as mock_download:
            yield mock_fetch, mock_download
            
@pytest.fixture
def parameters_model_tiny():
    return ModelState(
        id_model=1,
        model_name="tiny",
        enabled=True,
        status="queued_for_download",
        progress=0.0,
        error=None
    )

@pytest.fixture
def parameters_model_base():
    return ModelState(
        id_model=2,
        model_name="base",
        enabled=True,
        status="queued_for_download",
        progress=0.0,
        error=None
    )

@pytest.fixture
def models_rows():
    return [
        {"id_model": 1, "model_name": "tiny"},
        {"id_model": 2, "model_name": "base"},
    ]

@pytest.fixture
def enabled_rows():
    return [
        {"model_id": 1},
        {"model_id": 2},
    ]

@pytest_asyncio.fixture
async def app() -> FastAPI:
    return real_app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_audio_file(tmp_path: Path) -> Path:
    audio_path = tmp_path / "test_audio.mp3"
    audio_path.write_bytes(b"fake audio content")
    return audio_path


@pytest_asyncio.fixture
async def mock_app_state() -> AsyncGenerator[AppState, None]:
    mock_torch_cuda = MagicMock()
    mock_torch_cuda.set_device.return_value = None
    mock_torch_cuda.reset_peak_memory_stats.return_value = None
    mock_torch_cuda.max_memory_allocated.return_value = 4000 * 1024 * 1024
    mock_torch_cuda.empty_cache.return_value = None

    with patch("app.web_routes._core") as mock_core, \
            patch("app.queueing.gpu_count", return_value=1), \
            patch("app.queueing.torch_cuda_available", return_value=True), \
            patch("app.queueing.torch_cuda_device_count", return_value=1), \
            patch("app.model_registry.fetch_all") as mock_fetch_all, \
            patch("whisper.load_model") as mock_whisper_load, \
            patch("app.whisper_runner.gpu_metrics") as mock_gpu_metrics, \
            patch("app.queueing.httpx.AsyncClient") as mock_httpx_client, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess, \
            patch("torch.cuda", mock_torch_cuda):
        mock_fetch_all.side_effect = [
            [{"id_model": 1, "model_name": "base"}],
            [{"model_id": 1}]
        ]

        mock_preprocess.return_value = None

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Это тестовый текст транскрибации",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Это тестовый текст"}]
        }
        mock_whisper_load.return_value = mock_model

        mock_gpu_metrics.side_effect = [
            (50.0, 2000.0, 8192.0),
            (75.0, 3000.0, 8192.0),
            (60.0, 2500.0, 8192.0),
            (55.0, 2200.0, 8192.0)
        ]

        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__.return_value.post.return_value = MagicMock(status_code=200)
        mock_httpx_client.return_value = mock_client_instance

        state = AppState()
        await state.startup()

        state.models._models = {
            "base": ModelState(
                id_model=1,
                model_name="base",
                enabled=True,
                status="downloaded",
                progress=100.0,
                error=None
            )
        }

        def mock_is_model_known(name):
            return name == "base"

        state.models.is_model_known = mock_is_model_known

        max_wait = 10
        start_time = asyncio.get_event_loop().time()
        while state.health_status != "ready" and asyncio.get_event_loop().time() - start_time < max_wait:
            await asyncio.sleep(0.1)

        mock_core.return_value = state
        yield state
        await state.shutdown()

@pytest.fixture
def data_model_base():
    return {
        "model": "base",
        "callback_url": "http://test-callback.example/whisper-callback",
        "language": "Russian"
    }

@pytest.fixture
def data_model_base_en():
    return {
        "model": "base",
        "callback_url": "http://test-callback.example/whisper-callback",
        "language": "English"
    }

@pytest.fixture
def data_model_base_lng_none():
    return {
            "model": "base",
            "callback_url": "http://test-callback.example/whisper-callback",
            "language": None
        }

@pytest.fixture
def data_model_tiny():
    return {
        "model": "tiny",
        "language": "Russian",
        "callback_url": "http://test-callback.example/whisper-callback"
    }

@pytest.fixture
def files(test_audio_file):
    return {
        "file": (test_audio_file.name, test_audio_file.read_bytes(), "audio/mpeg")
    }

@pytest.fixture
def test_transcription_text():
    return {
            "text": "Это тестовый текст транскрибации",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Это тестовый текст"}]
        }

@pytest.fixture
async def client_is_endpoint():
    async with AsyncClient(base_url=BASE_URL, timeout=30.0) as ac:
        yield ac

@pytest_asyncio.fixture
async def mock_app_state_with_multiple_models() -> AsyncGenerator[AppState, None]:
    mock_torch_cuda = MagicMock()
    mock_torch_cuda.set_device.return_value = None
    mock_torch_cuda.reset_peak_memory_stats.return_value = None
    mock_torch_cuda.max_memory_allocated.return_value = 4000 * 1024 * 1024
    mock_torch_cuda.empty_cache.return_value = None

    with patch("app.web_routes._core") as mock_core, \
            patch("app.queueing.gpu_count", return_value=1), \
            patch("app.queueing.torch_cuda_available", return_value=True), \
            patch("app.queueing.torch_cuda_device_count", return_value=1), \
            patch("app.model_registry.fetch_all") as mock_fetch_all, \
            patch("torch.cuda", mock_torch_cuda):
        mock_fetch_all.side_effect = [
            [
                {"id_model": 1, "model_name": "tiny"},
                {"id_model": 2, "model_name": "base"},
                {"id_model": 3, "model_name": "small"}
            ],
            [{"model_id": 1}, {"model_id": 2}, {"model_id": 3}]
        ]

        state = AppState()
        await state.startup()

        state.models._models = {
            "tiny": ModelState(
                id_model=1,
                model_name="tiny",
                enabled=True,
                status="downloaded",
                progress=100.0,
                error=None
            ),
            "base": ModelState(
                id_model=2,
                model_name="base",
                enabled=True,
                status="downloaded",
                progress=100.0,
                error=None
            ),
            "small": ModelState(
                id_model=3,
                model_name="small",
                enabled=True,
                status="downloaded",
                progress=100.0,
                error=None
            )
        }

        def mock_is_model_known(name):
            return name in ["tiny", "base", "small"]

        state.models.is_model_known = mock_is_model_known

        max_wait = 10
        start_time = asyncio.get_event_loop().time()
        while state.health_status != "ready" and asyncio.get_event_loop().time() - start_time < max_wait:
            await asyncio.sleep(0.1)

        mock_core.return_value = state
        yield state
        await state.shutdown()

@pytest.fixture
def texts_by_language():
    return {
        "Russian": "Это тестовый текст на русском языке",
        "English": "This is a test text in English",
        "German": "Dies ist ein Test text auf Deutsch",
        "French": "Ceci est un text e de test en français"
    }

@pytest.fixture
def models_with_delays():
    return {
            "tiny": 0.5,
            "base": 1.0,
            "small": 2.0
        }
