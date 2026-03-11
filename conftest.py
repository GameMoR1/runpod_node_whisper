import pytest
from unittest.mock import AsyncMock, patch, MagicMock


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