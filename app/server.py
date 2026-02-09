from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.state import AppState
from app.web_routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    app.state.core = state
    await state.startup()
    try:
        yield
    finally:
        await state.shutdown()


app = FastAPI(lifespan=lifespan)
app.include_router(router)

