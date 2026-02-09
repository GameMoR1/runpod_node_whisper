from __future__ import annotations

import asyncio
import uuid
import logging
from pathlib import Path
from typing import Any, Optional

import anyio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect

from app.config import settings
from app.types import HealthStatus


logger = logging.getLogger("whisper_node.http")


router = APIRouter()


def _core(request: Request):
    return request.app.state.core


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    core = _core(request)
    status: HealthStatus = core.health_status
    return {"status": status}


@router.get("/queue")
async def queue_view(request: Request) -> dict[str, Any]:
    core = _core(request)
    q = core.queue
    if q is None:
        return {"status": "idle", "queued": [], "running": []}
    queued_ids, running_ids = q.snapshot_ids()
    if not queued_ids and not running_ids:
        return {"status": "idle", "queued": [], "running": []}
    return {"status": "busy", "queued": queued_ids, "running": running_ids}


@router.get("/status")
async def status(request: Request, job_id: str) -> dict[str, Any]:
    core = _core(request)
    q = core.queue
    if q is None:
        raise HTTPException(status_code=404, detail="job not found")
    job = q.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return q.serialize_job(job)


@router.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    callback_url: str = Form(...),
    language: Optional[str] = Form(None),
) -> dict[str, Any]:
    core = _core(request)
    if core.health_status != "ready":
        raise HTTPException(status_code=503, detail={"status": core.health_status})

    q = core.queue
    if q is None:
        raise HTTPException(status_code=503, detail="service not initialized")

    if not core.models or not core.models.is_model_known(model):
        raise HTTPException(status_code=400, detail="unknown model")

    job_id = str(uuid.uuid4())
    job_dir = Path(settings.UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    in_path = job_dir / "input"

    async with await anyio.open_file(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)

    lang = language or settings.WHISPER_DEFAULT_LANGUAGE
    await q.enqueue(job_id=job_id, model=model, language=lang, callback_url=callback_url, file_dir=str(job_dir))
    logger.info("transcribe accepted: %s model=%s language=%s", job_id, model, lang)
    return {"job_id": job_id}


@router.get("/dashboard/state")
async def dashboard_state(request: Request) -> dict[str, Any]:
    core = _core(request)
    models = core.models
    q = core.queue
    return {
        "health": {"status": core.health_status, "error": core.health_error},
        "models": [] if models is None else models.serialize_public(),
        "gpus": [] if q is None else q.serialize_gpus_public(),
        "jobs": {"total": 0, "queued": 0, "running": 0, "queued_ids": [], "running_ids": []}
        if q is None
        else q.serialize_jobs_public(),
        "refresh_ms": settings.DASHBOARD_REFRESH_MS,
    }


@router.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            core = websocket.app.state.core
            models = core.models
            q = core.queue
            await websocket.send_json(
                {
                    "health": {"status": core.health_status, "error": core.health_error},
                    "models": [] if models is None else models.serialize_public(),
                    "gpus": [] if q is None else q.serialize_gpus_public(),
                    "jobs": {
                        "total": 0,
                        "queued": 0,
                        "running": 0,
                        "queued_ids": [],
                        "running_ids": [],
                    }
                    if q is None
                    else q.serialize_jobs_public(),
                    "refresh_ms": settings.DASHBOARD_REFRESH_MS,
                }
            )
            await asyncio.sleep(max(0.25, settings.DASHBOARD_REFRESH_MS / 1000))
    except WebSocketDisconnect:
        return


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    html = _dashboard_html()
    return HTMLResponse(content=html)


def _dashboard_html() -> str:
    html = """<!doctype html>
<html lang=\"ru\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Whisper CUDA</title>
    <script src=\"https://unpkg.com/lucide@latest\"></script>
    <style>
      :root {
        --bg: #05070B;
        --panel: #0C1016;
        --card: #0F172A;
        --cardHover: #111827;
        --border: rgba(255,255,255,0.08);
        --text: #E5E7EB;
        --muted: #9CA3AF;
        --accent: #2563EB;
        --danger: #EF4444;
        --ok: #10B981;
        --warn: #F59E0B;
      }
      html, body { height: 100%; }
      body { margin: 0; background: var(--bg); color: var(--text); font-family: Inter, Roboto, system-ui, -apple-system, Segoe UI, Arial, sans-serif; }
      .wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }
      .top { display:flex; align-items:center; justify-content:space-between; gap: 12px; margin-bottom: 16px; }
      .brand { display:flex; align-items:center; gap: 10px; font-weight: 700; }
      .btn { background: var(--accent); border: 1px solid rgba(37,99,235,0.6); color: white; padding: 10px 14px; border-radius: 10px; cursor: pointer; font-weight: 600; }
      .btn:hover { filter: brightness(1.05); }
      .meta { color: var(--muted); font-size: 12px; }
      .grid { display:grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }
      .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
      .card:hover { background: var(--cardHover); }
      .kpi { grid-column: span 3; }
      .kpiTitle { color: var(--muted); font-size: 12px; margin-bottom: 6px; display:flex; gap:8px; align-items:center; }
      .kpiValue { font-size: 20px; font-weight: 700; }
      .sectionTitle { font-size: 14px; font-weight: 700; margin: 24px 0 12px; color: var(--text); }
      .chip { display:inline-flex; align-items:center; gap: 6px; padding: 4px 10px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); color: var(--muted); }
      .chip.ok { color: var(--ok); border-color: rgba(16,185,129,0.4); }
      .chip.err { color: var(--danger); border-color: rgba(239,68,68,0.4); }
      .chip.warn { color: var(--warn); border-color: rgba(245,158,11,0.4); }
      .cols2 { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }
      .list { display:flex; flex-direction:column; gap: 8px; max-height: 320px; overflow:auto; }
      .row { display:flex; align-items:center; justify-content:space-between; gap: 12px; padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: rgba(255,255,255,0.02); }
      .row:hover { background: rgba(255,255,255,0.04); }
      .iconBtn { width: 34px; height: 34px; display:inline-flex; align-items:center; justify-content:center; border-radius: 10px; border: 1px solid var(--border); background: transparent; color: var(--muted); cursor: pointer; }
      .iconBtn:hover { color: var(--text); border-color: rgba(255,255,255,0.16); }
      .bar { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow:hidden; border: 1px solid var(--border); }
      .bar > div { height: 100%; background: var(--accent); width: 0%; }
      .sub { color: var(--muted); font-size: 12px; }
      @media (max-width: 1024px) { .kpi { grid-column: span 6; } .cols2 { grid-template-columns: 1fr; } }
      @media (max-width: 640px) { .kpi { grid-column: span 12; } }
    </style>
  </head>
  <body>
    <div class=\"wrap\">
      <div class=\"top\">
        <div class=\"brand\">
          <i data-lucide=\"waveform\"></i>
          <div>Whisper CUDA</div>
        </div>
        <div style=\"display:flex; align-items:center; gap: 12px;\">
          <div class=\"meta\" id=\"updated\">—</div>
        </div>
      </div>

      <div class=\"grid\">
        <div class=\"card kpi\"><div class=\"kpiTitle\"><i data-lucide=\"activity\"></i>Health</div><div class=\"kpiValue\" id=\"health\">—</div></div>
        <div class=\"card kpi\"><div class=\"kpiTitle\"><i data-lucide=\"layers\"></i>Jobs total</div><div class=\"kpiValue\" id=\"jobsTotal\">0</div></div>
        <div class=\"card kpi\"><div class=\"kpiTitle\"><i data-lucide=\"list\"></i>Queued</div><div class=\"kpiValue\" id=\"jobsQueued\">0</div></div>
        <div class=\"card kpi\"><div class=\"kpiTitle\"><i data-lucide=\"cpu\"></i>Processing</div><div class=\"kpiValue\" id=\"jobsRunning\">0</div></div>
      </div>

      <div class=\"sectionTitle\">GPUs</div>
      <div class=\"grid\" id=\"gpuGrid\"></div>

      <div class=\"sectionTitle\">Models</div>
      <div class=\"card\"><div class=\"list\" id=\"modelList\"></div></div>

      <div class=\"sectionTitle\">Jobs</div>
      <div class=\"cols2\">
        <div class=\"card\"><div style=\"font-weight:700;\">Queued</div><div class=\"list\" id=\"queuedList\" style=\"margin-top: 12px;\"></div></div>
        <div class=\"card\"><div style=\"font-weight:700;\">Processing</div><div class=\"list\" id=\"runningList\" style=\"margin-top: 12px;\"></div></div>
      </div>
    </div>

    <script>
      function setText(id, v) { document.getElementById(id).textContent = v; }
      function chipClass(status) {
        if (status === 'ready') return 'chip ok';
        if (status === 'error') return 'chip err';
        return 'chip warn';
      }
      function modelChip(status) {
        if (status === 'downloaded') return 'chip ok';
        if (status === 'error') return 'chip err';
        if (status === 'downloading') return 'chip warn';
        return 'chip';
      }
      async function copyText(txt) {
        try { await navigator.clipboard.writeText(txt); } catch (e) {}
      }
      function rowItem(txt) {
        const el = document.createElement('div');
        el.className = 'row';
        const left = document.createElement('div');
        left.className = 'mono';
        left.textContent = txt;
        const btn = document.createElement('button');
        btn.className = 'iconBtn';
        btn.innerHTML = '<i data-lucide="copy"></i>';
        btn.onclick = () => copyText(txt);
        el.appendChild(left);
        el.appendChild(btn);
        return el;
      }
      function emptyItem(msg) {
        const el = document.createElement('div');
        el.className = 'sub';
        el.textContent = msg;
        return el;
      }
      function bar(percent) {
        const b = document.createElement('div');
        b.className = 'bar';
        const f = document.createElement('div');
        f.style.width = String(Math.max(0, Math.min(100, percent))) + '%';
        b.appendChild(f);
        return b;
      }
      function gpuCard(g) {
        const card = document.createElement('div');
        card.className = 'card';
        card.style.gridColumn = 'span 4';

        const title = document.createElement('div');
        title.style.display = 'flex';
        title.style.alignItems = 'center';
        title.style.justifyContent = 'space-between';
        title.style.gap = '12px';

        const tLeft = document.createElement('div');
        tLeft.style.fontWeight = '700';
        tLeft.textContent = 'GPU ' + g.index + ': ' + g.name;

        const chip = document.createElement('span');
        chip.className = g.status === 'running' ? 'chip warn' : 'chip ok';
        chip.textContent = g.status;

        title.appendChild(tLeft);
        title.appendChild(chip);
        card.appendChild(title);

        const line1 = document.createElement('div');
        line1.className = 'sub';
        line1.style.marginTop = '10px';
        line1.innerHTML = 'Model: <span class="mono">' + (g.current_model || '—') + '</span>';
        card.appendChild(line1);

        const line2 = document.createElement('div');
        line2.className = 'sub';
        line2.innerHTML = 'Job: <span class="mono">' + (g.current_job_id || '—') + '</span>';
        card.appendChild(line2);

        const u = document.createElement('div');
        u.className = 'sub';
        u.style.marginTop = '12px';
        u.textContent = 'GPU util: ' + Math.round(g.util_percent) + '%';
        card.appendChild(u);
        card.appendChild(bar(g.util_percent));

        const v = document.createElement('div');
        v.className = 'sub';
        v.style.marginTop = '12px';
        v.textContent = 'VRAM: ' + Math.round(g.vram_used_mb) + ' / ' + Math.round(g.vram_total_mb) + ' MB (' + Math.round(g.vram_used_percent) + '%)';
        card.appendChild(v);
        card.appendChild(bar(g.vram_used_percent));

        return card;
      }
      function renderList(containerId, items, emptyMsg) {
        const root = document.getElementById(containerId);
        root.innerHTML = '';
        if (!items || items.length === 0) {
          root.appendChild(emptyItem(emptyMsg));
          return;
        }
        for (const it of items) root.appendChild(rowItem(it));
      }
      function renderModels(items) {
        const root = document.getElementById('modelList');
        root.innerHTML = '';
        if (!items || items.length === 0) {
          root.appendChild(emptyItem('No enabled models'));
          return;
        }
        for (const m of items) {
          const row = document.createElement('div');
          row.className = 'row';
          const left = document.createElement('div');
          left.innerHTML = '<div style="font-weight:700">' + m.model_name + '</div><div class="sub mono">id_model: ' + m.id_model + '</div>';
          const right = document.createElement('div');
          right.style.display = 'flex';
          right.style.flexDirection = 'column';
          right.style.alignItems = 'flex-end';
          right.style.gap = '6px';
          const chip = document.createElement('span');
          chip.className = modelChip(m.status);
          chip.textContent = m.status;
          right.appendChild(chip);
          if (m.status === 'downloading') {
            const p = document.createElement('div');
            p.className = 'sub';
            p.textContent = String(Math.round(m.progress)) + '%';
            right.appendChild(p);
          }
          row.appendChild(left);
          row.appendChild(right);
          root.appendChild(row);
        }
      }
      function renderGpus(items) {
        const root = document.getElementById('gpuGrid');
        root.innerHTML = '';
        if (!items || items.length === 0) {
          const card = document.createElement('div');
          card.className = 'card';
          card.style.gridColumn = 'span 12';
          card.appendChild(emptyItem('No NVIDIA GPUs detected'));
          root.appendChild(card);
          return;
        }
        for (const g of items) root.appendChild(gpuCard(g));
      }
      function applyState(data) {
        const hs = (data.health && data.health.status) ? data.health.status : 'starting';
        const he = (data.health && data.health.error) ? String(data.health.error) : '';
        const h = document.getElementById('health');
        h.innerHTML = '<span class="' + chipClass(hs) + '"' + (he ? (' title="' + he.replaceAll('"', '\\"') + '"') : '') + '>' + hs + '</span>';
        const jobs = data.jobs || {};
        setText('jobsTotal', String(jobs.total || 0));
        setText('jobsQueued', String(jobs.queued || 0));
        setText('jobsRunning', String(jobs.running || 0));
        renderGpus(data.gpus || []);
        renderModels(data.models || []);
        renderList('queuedList', jobs.queued_ids || [], 'Queue is empty');
        renderList('runningList', jobs.running_ids || [], 'No running jobs');
        const now = new Date();
        document.getElementById('updated').textContent = 'Last updated: ' + now.toLocaleTimeString();
        lucide.createIcons();
        return data.refresh_ms || __REFRESH_MS__;
      }

      async function loadState() {
        const res = await fetch('/dashboard/state', { cache: 'no-store' });
        const data = await res.json();
        return applyState(data);
      }
      let timer = null;
      async function startPolling() {
        if (timer) clearInterval(timer);
        const ms = await loadState();
        timer = setInterval(loadState, ms);
      }

      function startWebSocket() {
        const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
        const url = proto + '://' + location.host + '/ws/dashboard';
        let ws = null;
        let backoffMs = 500;

        function connect() {
          try {
            ws = new WebSocket(url);
          } catch (e) {
            startPolling();
            scheduleReconnect();
            return;
          }

          ws.onopen = () => {
            if (timer) { clearInterval(timer); timer = null; }
            backoffMs = 500;
          };
          ws.onmessage = (ev) => {
            try {
              const data = JSON.parse(ev.data);
              applyState(data);
            } catch (e) {}
          };
          ws.onerror = () => {
            try { ws.close(); } catch (e) {}
          };
          ws.onclose = () => {
            startPolling();
            scheduleReconnect();
          };
        }

        function scheduleReconnect() {
          const ms = Math.min(10_000, backoffMs);
          backoffMs = Math.min(10_000, Math.floor(backoffMs * 1.6));
          setTimeout(connect, ms);
        }

        connect();
      }
      lucide.createIcons();
      startWebSocket();
    </script>
  </body>
</html>"""
    return html.replace("__REFRESH_MS__", str(settings.DASHBOARD_REFRESH_MS))
