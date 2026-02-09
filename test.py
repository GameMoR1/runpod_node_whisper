from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import httpx


BASE_URL = "https://opkxvi3839vreq-8000.proxy.runpod.net"
AUDIO_PATH = Path(r"C:\Users\akorn\Documents\GitHub\runpod_node_whisper\GMT20260206-114328_Recording.m4a")
TIMEOUT_S = 900
POLL_S = 2.0
MODEL = "small"
LANGUAGE = "ru"
REQUESTS = 5


def _pick_model_from_state(state: dict[str, Any]) -> Optional[str]:
    models = state.get("models")
    if not isinstance(models, list) or not models:
        return None
    ready = [m for m in models if isinstance(m, dict) and m.get("status") in {"downloaded", "ready"}]
    if ready:
        m0 = ready[0]
        name = m0.get("model_name")
        return str(name) if name else None
    m0 = models[0]
    name = m0.get("model_name") if isinstance(m0, dict) else None
    return str(name) if name else None


def _print_state_summary(state: dict[str, Any]) -> None:
    health = state.get("health") if isinstance(state.get("health"), dict) else {}
    hs = health.get("status")
    he = health.get("error")
    print(f"Health: {hs}")
    if he:
        print(f"Health error: {he}")

    models = state.get("models")
    if isinstance(models, list) and models:
        print("Models:")
        for m in models:
            if not isinstance(m, dict):
                continue
            print(f"- {m.get('model_name')} (status={m.get('status')}, enabled={m.get('enabled')})")
    else:
        print("Models: (empty)")


def main() -> int:
    base_url = BASE_URL[:-1] if BASE_URL.endswith("/") else BASE_URL
    audio_path = AUDIO_PATH
    if not audio_path.exists() or not audio_path.is_file():
        print(f"Файл не найден: {audio_path}")
        return 2

    timeout_s = int(TIMEOUT_S)
    poll_s = float(POLL_S)

    transcribe_url = f"{base_url}/transcribe"
    state_url = f"{base_url}/dashboard/state"
    status_url = f"{base_url}/status"

    with httpx.Client(timeout=60.0) as client:
        state: Optional[dict[str, Any]] = None
        try:
            r = client.get(state_url, headers={"Cache-Control": "no-cache"})
            r.raise_for_status()
            state = r.json()
        except Exception as e:
            print(f"Не удалось получить состояние дашборда ({state_url}): {e}")

        if isinstance(state, dict):
            _print_state_summary(state)

        model = MODEL
        language = LANGUAGE

        callback_url = "http://127.0.0.1:9/callback"
        data = {"model": model, "callback_url": callback_url, "language": language}

        for i in range(REQUESTS):
            print(f"\n=== REQUEST {i + 1}/{REQUESTS} ===")
            print("Отправляю запрос...")

            with audio_path.open("rb") as f:
                files = {"file": (audio_path.name, f, "application/octet-stream")}
                try:
                    r = client.post(transcribe_url, data=data, files=files)
                except Exception as e:
                    print(f"POST {transcribe_url} не удался: {e}")
                    return 3

            if r.status_code != 200:
                try:
                    print(f"Ошибка {r.status_code}: {r.json()}")
                except Exception:
                    print(f"Ошибка {r.status_code}: {r.text}")
                return 4

            payload = r.json()
            job_id = payload.get("job_id") if isinstance(payload, dict) else None
            if not job_id:
                print(f"Неожиданный ответ: {payload}")
                return 5

            print(f"job_id: {job_id}")
            print("Ожидаю завершения (через /status)...")

            deadline = time.time() + float(timeout_s)
            last_status = None
            while time.time() < deadline:
                try:
                    rs = client.get(status_url, params={"job_id": job_id}, headers={"Cache-Control": "no-cache"})
                    rs.raise_for_status()
                    st = rs.json()
                except Exception as e:
                    print(f"Ошибка опроса /status: {e}")
                    time.sleep(poll_s)
                    continue

                if isinstance(st, dict):
                    status = st.get("status")
                    if status != last_status:
                        print(f"status: {status}")
                        last_status = status

                    if status in {"completed", "failed"}:
                        if status == "failed":
                            print(f"error: {st.get('error')}")
                            return 10
                        result = st.get("result") if isinstance(st.get("result"), dict) else {}
                        text = result.get("text") if isinstance(result, dict) else None
                        if text:
                            print("\n=== TEXT ===")
                            print(text)
                        break

                time.sleep(poll_s)

            if time.time() >= deadline:
                print("Таймаут ожидания результата")
                return 11

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
