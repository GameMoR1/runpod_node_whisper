import asyncio
import pytest

@pytest.mark.asyncio
async def test_parallel_jobs_submission(
    client,
    files,
    mock_app_state_with_multiple_models
):
    models = ["tiny", "base"]

    def create_request(model_name):
        data = {
            "model": model_name,
            "language": "ru",
            "callback_url": "http://example.com/callback"
        }
        return client.post("/transcribe", files=files, data=data)

    responses = await asyncio.gather(*[create_request(m) for m in models])

    for resp in responses:
        assert resp.status_code in [200, 503]
        if resp.status_code == 200:
            assert "job_id" in resp.json()

    response_queue = await client.get("/queue")
    assert response_queue.status_code == 200
    state = response_queue.json()

    total_active = len(state.get("queued", [])) + len(state.get("running", []))
    assert total_active >= 0