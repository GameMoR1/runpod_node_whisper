# Whisper CUDA FastAPI Service

FastAPI-сервис для CUDA-транскрибации на базе `openai-whisper` с очередью задач (1 GPU = 1 воркер), загрузкой enabled-моделей из БД и callback по завершению.

## Требования

- Python 3.11–3.12 (Windows + Python 3.13 не поддерживается для установки зависимостей этого проекта)
- NVIDIA GPU + драйвер
- CUDA runtime (рекомендуется CUDA 11.8)
- `ffmpeg` доступен в PATH

## Установка

1) Создай виртуальное окружение:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

2) Установка одной командой (рекомендуется):

```bash
pip install -r requirements-all.txt
```

Если `torch.version.cuda` показывает `None`, значит поставился CPU-only PyTorch — удали `torch/torchvision/torchaudio` и поставь заново (или используй `--force-reinstall`).

Проверка CUDA:

```bash
python -c "import torch; print('cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
```

Альтернатива (если хочешь ставить по шагам):

```bash
pip install -r requirements.txt
pip install -r requirements-torch-cu118.txt
pip install --no-build-isolation -r requirements-whisper.txt
```

Примечание: на Windows рекомендуется Python 3.11–3.12. На Python 3.13 часть зависимостей может потребовать сборку из исходников.

4) Проверь `.env`.

Минимально нужно:

```
DB_HOST=...
DB_PORT=...
DB_NAME=...
DB_USER=...
DB_PASS=...
```

Используется PostgreSQL.

## Запуск (без логирования access log)

```bash
python run.py
```

На Windows для асинхронного PostgreSQL-драйвера используется `WindowsSelectorEventLoopPolicy` (это уже выставляется в `run.py`).

По умолчанию сервис стартует на `0.0.0.0:8000`.

## Логика загрузки моделей

При старте сервис:

1) Берёт из таблицы `whisper_models` все строки, где `source='whisper'`.
2) Для найденных `id_model` проверяет, что в `model_settings` есть записи с `enabled=true`.
3) Модели, которые включены, скачиваются на диск в `data/whisper_models` (в VRAM не держатся).

Если таблицы используют другие названия ключей (`model_id` вместо `id_model`), сервис делает fallback-запросы.

## Очередь и параллелизм

- Очередь в памяти процесса.
- Количество параллельных воркеров = количеству NVIDIA GPU.
- 1 GPU выполняет максимум 1 задачу одновременно.

## API

### GET /health

Возвращает состояние готовности:

```json
{ "status": "starting" }
```

Статусы:

- `starting`: модели скачиваются/инициализация
- `ready`: сервис готов принимать задачи
- `error`: критическая ошибка

### POST /transcribe

`multipart/form-data`:

- `file` (обязательно): любой аудио/видео формат
- `model` (обязательно): имя whisper-модели (должно быть среди enabled из БД)
- `callback_url` (обязательно): URL для callback POST
- `language` (опционально): по умолчанию `Russian`

Response:

```json
{ "job_id": "..." }
```

Перед транскрибацией файл предобрабатывается через `ffmpeg`:

`-ac 1 -ar 16000 -af silenceremove=start_periods=1:start_silence=0.5:start_threshold=-40dB`

### GET /queue

Если очередь пуста и нет задач в обработке:

```json
{ "status": "idle", "queued": [], "running": [] }
```

Если есть задачи:

```json
{ "status": "busy", "queued": ["..."], "running": ["..."] }
```

### GET /status?job_id=...

Возвращает статус задачи и (если готово) результат.

### Callback payload

После завершения обработки сервис делает `POST` на `callback_url`.

Успех:

```json
{
  "job_id": "...",
  "status": "completed",
  "model": "base",
  "language": "Russian",
  "queue_time_s": 0.2,
  "processing_time_s": 12.4,
  "result": {
    "text": "...",
    "segments": [
      {"start": 0.0, "end": 2.0, "text": "..."}
    ],
    "queue_time_s": 0.2,
    "processing_time_s": 12.4,
    "gpu": {
      "index": 0,
      "util_avg_percent": 63.5,
      "util_max_percent": 98.0,
      "vram_total_mb": 24576.0,
      "vram_used_avg_mb": 4200.0,
      "vram_used_max_mb": 6100.0,
      "vram_used_percent": 17.0,
      "vram_used_percent_max": 24.8
    },
    "token_count": 1234
  },
  "error": null,
  "callback": {"delivered": true, "delivered_at_ms": 1700000000000, "error": null}
}
```

Ошибка:

```json
{
  "job_id": "...",
  "status": "failed",
  "result": null,
  "error": "..."
}
```

## Dashboard

Открой `http://localhost:8000/`.

На странице показывается:

- `health` (starting/ready/error)
- список enabled-моделей и статус скачивания
- список GPU: util/VRAM (used/total/%), текущий `job_id` и модель на GPU
- счётчики задач и списки `job_id` (queue/processing)

Детали задач (callback_url, тексты, ошибки, результаты) на дашборде не отображаются.

Данные обновляются автоматически через WebSocket `GET /ws/dashboard` (fallback: polling `GET /dashboard/state`).
