import asyncio
from pathlib import Path

import pytest

from httpx import AsyncClient
from app.state import AppState
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_upload_file_error(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state: AppState,
        data_model_base,
        files
):
    async def failing_file_operation(*args, **kwargs):
        raise Exception("File upload failed")

    with patch("app.web_routes.anyio.open_file", side_effect=failing_file_operation):
        with pytest.raises(Exception) as exc_info:
            await client.post("/transcribe", data=data_model_base, files=files)

        assert "File upload failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_transcription_error(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state: AppState,
        data_model_base,
        files
):
    with patch("whisper.load_model") as mock_whisper_load, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess:

        mock_preprocess.return_value = None

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_whisper_load.return_value = mock_model

        response = await client.post("/transcribe", data=data_model_base, files=files)
        assert response.status_code == 200
        response_data = response.json()
        assert "job_id" in response_data
        job_id = response_data["job_id"]

        max_wait = 30
        start_time = asyncio.get_event_loop().time()
        final_status = None

        while asyncio.get_event_loop().time() - start_time < max_wait:
            status_response = await client.get(f"/status?job_id={job_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()

            if status_data["status"] in ["completed", "failed"]:
                final_status = status_data
                break

            await asyncio.sleep(0.5)

        assert final_status is not None
        assert final_status["status"] == "failed"
        assert final_status["error"] is not None
        assert "Transcription failed" in final_status["error"]
        assert final_status["result"] is None


@pytest.mark.asyncio
async def test_callback_error(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state: AppState,
        data_model_base,
        files,
        test_transcription_text
):
    with patch("whisper.load_model") as mock_whisper_load, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess, \
            patch("app.queueing.httpx.AsyncClient") as mock_httpx_client:

        mock_preprocess.return_value = None

        mock_model = MagicMock()
        mock_model.transcribe.return_value = test_transcription_text
        mock_whisper_load.return_value = mock_model

        mock_client_instance = MagicMock()
        mock_post = MagicMock()
        mock_post.side_effect = Exception("Callback failed")
        mock_client_instance.__aenter__.return_value.post = mock_post
        mock_httpx_client.return_value = mock_client_instance

        response = await client.post("/transcribe", data=data_model_base, files=files)
        assert response.status_code == 200
        response_data = response.json()
        assert "job_id" in response_data
        job_id = response_data["job_id"]

        max_wait = 30
        start_time = asyncio.get_event_loop().time()
        final_status = None

        while asyncio.get_event_loop().time() - start_time < max_wait:
            status_response = await client.get(f"/status?job_id={job_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()

            if status_data["status"] in ["completed", "failed"]:
                final_status = status_data
                break

            await asyncio.sleep(0.5)

        assert final_status is not None
        assert final_status["status"] == "completed"
        assert final_status["callback"]["delivered"] is False
        assert final_status["callback"]["error"] is not None
        assert "Callback failed" in final_status["callback"]["error"]


@pytest.mark.asyncio
async def test_callback_http_error(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state: AppState,
        data_model_base,
        files,
        test_transcription_text
):
    with patch("whisper.load_model") as mock_whisper_load, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess, \
            patch("app.queueing.httpx.AsyncClient") as mock_httpx_client:

        mock_preprocess.return_value = None

        mock_model = MagicMock()
        mock_model.transcribe.return_value = test_transcription_text
        mock_whisper_load.return_value = mock_model

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__.return_value.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        response = await client.post("/transcribe", data=data_model_base, files=files)
        assert response.status_code == 200
        response_data = response.json()
        assert "job_id" in response_data
        job_id = response_data["job_id"]

        max_wait = 30
        start_time = asyncio.get_event_loop().time()
        final_status = None

        while asyncio.get_event_loop().time() - start_time < max_wait:
            status_response = await client.get(f"/status?job_id={job_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()

            if status_data["status"] in ["completed", "failed"]:
                final_status = status_data
                break

            await asyncio.sleep(0.5)

        assert final_status is not None
        assert final_status["status"] == "completed"
        assert final_status["callback"]["delivered"] is False
        assert final_status["callback"]["error"] is not None