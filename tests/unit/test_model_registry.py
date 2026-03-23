import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import asyncio
from app.model_registry import ModelRegistry
from conftest import parameters_model_tiny

@pytest.mark.asyncio
async def test_load_enabled_models_from_db(mock_db_models):
    mock_fetch, mock_download = mock_db_models
    registry = ModelRegistry()
    await registry.load_from_db_and_prepare()

    all_models = registry.all()
    assert len(all_models) == 2

    assert "tiny" in registry._models
    assert "small" in registry._models
    assert "base" not in registry._models
    assert "medium" not in registry._models

    tiny_model = registry._models.get("tiny")
    assert tiny_model.id_model == 1
    assert tiny_model.model_name == "tiny"
    assert tiny_model.enabled == True

    small_model = registry._models.get("small")
    assert small_model.id_model == 3
    assert small_model.model_name == "small"
    assert small_model.enabled == True

    assert mock_fetch.call_count == 2
    assert mock_download.call_count == 2

@pytest.mark.asyncio
async def test_filter_only_enabled_models(mock_db_models):
    registry = ModelRegistry()
    await registry.load_from_db_and_prepare()

    enabled_count = 0
    for model_name, model_state in registry._models.items():
        assert model_state.enabled == True
        enabled_count += 1

    assert enabled_count == 2
    assert len(registry._models) == 2

@pytest.mark.asyncio
async def test_successful_model_download(parameters_model_tiny):
    registry = ModelRegistry()

    registry._models["tiny"] = parameters_model_tiny

    assert registry._models["tiny"].status == "queued_for_download"
    assert registry._models["tiny"].progress == 0.0

    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = MagicMock()

    with patch.dict('sys.modules', {'whisper': mock_whisper}):
        await registry._download_model("tiny")

    mock_whisper.load_model.assert_called_once_with(
        "tiny",
        device="cpu",
        download_root="data/whisper_models"
    )

    assert registry._models["tiny"].status == "downloaded"
    assert registry._models["tiny"].progress == 100.0
    assert registry._models["tiny"].error is None

@pytest.mark.asyncio
async def test_model_download_error_with_retry(parameters_model_tiny):
    registry = ModelRegistry()

    registry._models["tiny"] = parameters_model_tiny

    assert registry._models["tiny"].status == "queued_for_download"

    mock_whisper = MagicMock()
    mock_whisper.load_model.side_effect = Exception("Connection error")

    with patch.dict('sys.modules', {'whisper': mock_whisper}):
        await registry._download_model("tiny")

    assert mock_whisper.load_model.call_count == 3
    assert registry._models["tiny"].status == "error"
    assert registry._models["tiny"].progress == 0.0
    assert registry._models["tiny"].error == "Connection error"

@pytest.mark.asyncio
async def test_parallel_download_two_models(parameters_model_tiny, parameters_model_base):
    registry = ModelRegistry()

    registry._models["tiny"] = parameters_model_tiny

    registry._models["base"] = parameters_model_base

    mock_whisper = MagicMock()

    def side_effect(model_name, device, download_root):
        pass

    mock_whisper.load_model.side_effect = side_effect

    with patch.dict('sys.modules', {'whisper': mock_whisper}):
        with patch('app.model_registry.settings.MODEL_DOWNLOAD_ATTEMPTS', 1):
            with patch('app.model_registry.settings.MODEL_CACHE_DIR', "data/whisper_models"):
                with patch('app.model_registry.asyncio.sleep', return_value=None):
                    await asyncio.gather(
                        registry._download_model("tiny"),
                        registry._download_model("base")
                    )

    assert mock_whisper.load_model.call_count == 2
    mock_whisper.load_model.assert_any_call("tiny", device="cpu", download_root="data/whisper_models")
    mock_whisper.load_model.assert_any_call("base", device="cpu", download_root="data/whisper_models")

    assert registry._models["tiny"].status == "downloaded"
    assert registry._models["tiny"].progress == 100.0
    assert registry._models["tiny"].error is None

    assert registry._models["base"].status == "downloaded"
    assert registry._models["base"].progress == 100.0
    assert registry._models["base"].error is None

@pytest.mark.asyncio
async def test_skip_already_downloaded_model(models_rows, enabled_rows):

    with patch("app.model_registry.fetch_all", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = [models_rows, enabled_rows]

        with patch("app.model_registry.ModelRegistry._download_model") as mock_download:
            mock_download = AsyncMock()

            with patch("app.model_registry.ModelRegistry.is_model_known") as mock_is_known:
                mock_is_known.return_value = True

                registry = ModelRegistry()
                await registry.load_from_db_and_prepare()

    assert mock_fetch.call_count == 2
    assert mock_download.call_count == 0