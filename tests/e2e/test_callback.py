import asyncio
from pathlib import Path

import pytest
from httpx import AsyncClient
from app.state import AppState

@pytest.mark.asyncio
async def test_full_scenario_upload_to_callback(
        client: AsyncClient,
        test_audio_file: Path,
        mock_app_state: AppState,
        data_model_base,
        files
):

    response = await client.post("/transcribe", data=data_model_base, files=files)
    assert response.status_code == 200
    response_data = response.json()
    assert "job_id" in response_data
    job_id = response_data["job_id"]

    queue_response = await client.get("/queue")
    assert queue_response.status_code == 200
    queue_data = queue_response.json()
    assert queue_data["status"] == "busy"
    assert job_id in queue_data["queued"]

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
    assert final_status["job_id"] == job_id
    assert final_status["model"] == "base"
    assert final_status["language"] == "Russian"
    assert "queue_time_s" in final_status
    assert "processing_time_s" in final_status
    assert final_status["queue_time_s"] >= 0
    assert final_status["processing_time_s"] > 0

    assert final_status["result"] is not None
    assert "text" in final_status["result"]
    assert final_status["result"]["text"] == "Это тестовый текст транскрибации"
    assert "segments" in final_status["result"]
    assert len(final_status["result"]["segments"]) == 1
    assert "token_count" in final_status["result"]

    assert "gpu" in final_status["result"]
    gpu_metrics = final_status["result"]["gpu"]
    assert gpu_metrics["index"] == 0
    assert gpu_metrics["util_avg_percent"] > 0
    assert gpu_metrics["util_max_percent"] > 0
    assert gpu_metrics["vram_total_mb"] > 0
    assert gpu_metrics["vram_used_avg_mb"] > 0
    assert gpu_metrics["vram_used_max_mb"] > 0

    assert final_status["callback"]["delivered"] is True
    assert final_status["callback"]["delivered_at_ms"] is not None
    assert final_status["callback"]["error"] is None

    queue_after_response = await client.get("/queue")
    assert queue_after_response.status_code == 200
    queue_after_data = queue_after_response.json()
    assert queue_after_data["status"] == "idle"
    assert job_id not in queue_after_data.get("queued", [])
    assert job_id not in queue_after_data.get("running", [])

    health_response = await client.get("/health")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert health_data["status"] == "ready"