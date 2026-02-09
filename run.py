import asyncio
import os
import sys

import uvicorn

from app.logging_setup import setup_logging


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    setup_logging()
    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        access_log=True,
        log_config=None,
        server_header=False,
        date_header=False,
    )


if __name__ == "__main__":
    main()
