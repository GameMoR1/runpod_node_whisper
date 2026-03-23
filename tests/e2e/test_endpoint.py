import pytest
from app.server import app
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_endpoints_availability(client_is_endpoint, data_model_tiny, files):
    response = await client_is_endpoint.get("/health")
    assert response.status_code == 200

    response = await client_is_endpoint.get("/queue")
    assert response.status_code == 200

    response = await client_is_endpoint.get("/status", params={"job_id": "non-existent"})
    assert response.status_code == 404

    response = await client_is_endpoint.post("/transcribe", files=files, data=data_model_tiny)
    assert response.status_code in [200, 503]

    response = await client_is_endpoint.get("/dashboard/state")
    assert response.status_code == 200


def test_websocket_dashboard(mock_app_state_with_multiple_models):
    with TestClient(app) as tc:
        with tc.websocket_connect("/ws/dashboard") as ws:
            data = ws.receive_json()
            assert "models" in data
            assert "health" in data