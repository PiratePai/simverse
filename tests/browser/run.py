"""Run the workbench browser tests against a temporary Flask server."""
import os
from pathlib import Path
import subprocess
import sys
import threading

from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simverse.webapp import app  # noqa: E402


def main() -> int:
    server = make_server("127.0.0.1", 0, app, threaded=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        return subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("workbench.test.mjs"))],
            cwd=ROOT, env={**os.environ, "SIMVERSE_TEST_URL": f"http://127.0.0.1:{server.port}"},
            timeout=90,
        ).returncode
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
