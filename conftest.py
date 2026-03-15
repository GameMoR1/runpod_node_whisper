import pytest
from unittest.mock import AsyncMock, patch
from app.types import ModelState


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