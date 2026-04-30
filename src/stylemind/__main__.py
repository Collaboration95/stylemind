from __future__ import annotations

import os
import sys
import threading
import time
import uuid

import httpx
import uvicorn
from dotenv import load_dotenv

from stylemind.cli.chat import ChatCLI

_DEFAULT_PORT = 8000
_HEALTH_TIMEOUT = 30
_HEALTH_POLL_INTERVAL = 0.5


def _start_server(port: int) -> None:
    uvicorn.run("stylemind.main:create_app", factory=True, host="0.0.0.0", port=port, log_level="error")


def _wait_for_server(port: int, timeout: int = _HEALTH_TIMEOUT) -> bool:
    """Poll /health until ready or timeout."""
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                return True
        except httpx.ConnectError, httpx.TimeoutException, httpx.ReadError:
            pass
        time.sleep(_HEALTH_POLL_INTERVAL)
    return False


def main() -> None:
    load_dotenv()
    port = int(os.environ.get("SERVER_PORT", str(_DEFAULT_PORT)))

    if _wait_for_server(port, timeout=2):
        print(f"Connected to existing StyleMind server on port {port}.")
    else:
        thread = threading.Thread(target=_start_server, args=(port,), daemon=True)
        thread.start()
        print(f"Starting StyleMind server on port {port}...")
        if not _wait_for_server(port):
            print(f"ERROR: Server did not start within {_HEALTH_TIMEOUT}s.", file=sys.stderr)
            sys.exit(1)

    user_id = uuid.uuid4().hex[:8]
    base_url = f"http://localhost:{port}"

    cli = ChatCLI(base_url=base_url, user_id=user_id)
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
        sys.exit(0)


if __name__ == "__main__":
    main()
