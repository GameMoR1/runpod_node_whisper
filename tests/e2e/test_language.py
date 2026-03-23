import asyncio
from pathlib import Path

import pytest
from httpx import AsyncClient
from app.state import AppState
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_russian_on_tiny_model(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state_with_multiple_models: AppState,
        files,
        data_model_tiny
):
    russian_text = "Это тестовый текст на русском языке"

    with patch("whisper.load_model") as mock_whisper_load, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess, \
            patch("app.whisper_runner.postprocess_text") as mock_postprocess:

        mock_preprocess.return_value = None
        mock_postprocess.return_value = russian_text

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": russian_text,
            "segments": [{"start": 0.0, "end": 2.0, "text": russian_text}]
        }
        mock_whisper_load.return_value = mock_model

        response = await client.post("/transcribe", data=data_model_tiny, files=files)
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
        assert final_status["model"] == "tiny"
        assert final_status["language"] == "Russian"
        assert final_status["result"] is not None
        assert final_status["result"]["text"] == russian_text
        assert "segments" in final_status["result"]
        assert len(final_status["result"]["segments"]) == 1


@pytest.mark.asyncio
async def test_english_on_base_model(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state_with_multiple_models: AppState,
        files,
        data_model_base_en
):
    english_text = "This is a test text in English"

    with patch("whisper.load_model") as mock_whisper_load, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess, \
            patch("app.whisper_runner.postprocess_text") as mock_postprocess:

        mock_preprocess.return_value = None
        mock_postprocess.return_value = english_text

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": english_text,
            "segments": [{"start": 0.0, "end": 2.0, "text": english_text}]
        }
        mock_whisper_load.return_value = mock_model

        response = await client.post("/transcribe", data=data_model_base_en, files=files)
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
        assert final_status["model"] == "base"
        assert final_status["language"] == "English"
        assert final_status["result"] is not None
        assert final_status["result"]["text"] == english_text


@pytest.mark.asyncio
async def test_multilingual_detection(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state_with_multiple_models: AppState,
        texts_by_language,
        files
):

    with patch("whisper.load_model") as mock_whisper_load, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess, \
            patch("app.whisper_runner.postprocess_text") as mock_postprocess:

        mock_preprocess.return_value = None

        def transcribe_side_effect(*args, **kwargs):
            language = kwargs.get("language", "Russian")
            text = texts_by_language.get(language, "Unknown text")
            return {
                "text": text,
                "segments": [{"start": 0.0, "end": 2.0, "text": text}]
            }

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = transcribe_side_effect
        mock_whisper_load.return_value = mock_model
        mock_postprocess.side_effect = lambda x: x

        for language, expected_text in texts_by_language.items():

            data = {
                "model": "small",
                "callback_url": "http://test-callback.example/whisper-callback",
                "language": language
            }

            response = await client.post("/transcribe", data=data, files=files)
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
            assert final_status["model"] == "small"
            assert final_status["language"] == language
            assert final_status["result"] is not None
            assert final_status["result"]["text"] == expected_text


@pytest.mark.asyncio
async def test_auto_language_detection(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state_with_multiple_models: AppState,
        files,
        data_model_base_lng_none
):
    detected_text = "Это текст с автоопределением языка"

    with patch("whisper.load_model") as mock_whisper_load, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess, \
            patch("app.whisper_runner.postprocess_text") as mock_postprocess:

        mock_preprocess.return_value = None
        mock_postprocess.return_value = detected_text

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": detected_text,
            "segments": [{"start": 0.0, "end": 2.0, "text": detected_text}]
        }
        mock_whisper_load.return_value = mock_model

        response = await client.post("/transcribe", data=data_model_base_lng_none, files=files)
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
        assert final_status["model"] == "base"
        assert final_status["language"] == "Russian"
        assert final_status["result"] is not None
        assert final_status["result"]["text"] == detected_text


@pytest.mark.asyncio
async def test_model_size_impact_on_performance(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state_with_multiple_models: AppState,
        files,
        models_with_delays
):
    processing_times = {}

    with patch("whisper.load_model") as mock_whisper_load, \
            patch("app.queueing.preprocess_to_wav") as mock_preprocess, \
            patch("app.whisper_runner.gpu_metrics") as mock_gpu_metrics:

        mock_preprocess.return_value = None
        mock_gpu_metrics.return_value = (50.0, 2000.0, 8192.0)

        def create_mock_model(processing_delay: float):
            def delayed_transcribe(*args, **kwargs):
                import time
                time.sleep(processing_delay)
                return {
                    "text": "Test text",
                    "segments": [{"start": 0.0, "end": 2.0, "text": "Test text"}]
                }

            mock_model = MagicMock()
            mock_model.transcribe.side_effect = delayed_transcribe
            return mock_model

        for model_name, delay in models_with_delays.items():
            mock_whisper_load.return_value = create_mock_model(delay)

            data = {
                "model": model_name,
                "callback_url": "http://test-callback.example/whisper-callback",
                "language": "Russian"
            }

            response = await client.post("/transcribe", data=data, files=files)
            assert response.status_code == 200
            response_data = response.json()
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
            processing_times[model_name] = final_status["processing_time_s"]

    assert processing_times["tiny"] < processing_times["base"]
    assert processing_times["base"] < processing_times["small"]