FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ffmpeg -version >/dev/null \
 && ffprobe -version >/dev/null

COPY requirements.txt /app/requirements.txt
COPY constraints.txt /app/constraints.txt
RUN python -m pip install --upgrade pip \
 && pip install -c /app/constraints.txt -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python", "run.py"]

