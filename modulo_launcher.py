"""Standalone entry point for Modulo.

Starts the Flask server on a free local port, then opens Modulo in a native
window (pywebview) or, if that is unavailable, the default browser.
Works both from source (`python modulo_launcher.py`) and as a frozen app.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

class _State:  # shared with the heartbeat route
    @property
    def last_ping(self):
        flask_app = getattr(self, "flask", None)
        return flask_app.extensions.get("last_ping", 0) if flask_app else 0


app_state = _State()
FROZEN = getattr(sys, "frozen", False)
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def configure_bundled_tesseract() -> None:
    """Point pytesseract at the Tesseract copy shipped inside the app, if any."""
    exe = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    folder = BASE / "tesseract"
    binary = folder / exe
    if binary.exists() and not os.getenv("TESSERACT_CMD"):
        os.environ["TESSERACT_CMD"] = str(binary)
        os.environ.setdefault("TESSDATA_PREFIX", str(folder / "tessdata"))
        if sys.platform == "darwin":  # bundled dylibs sit next to the binary
            os.environ["DYLD_LIBRARY_PATH"] = f"{folder}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
        elif sys.platform.startswith("linux"):
            os.environ["LD_LIBRARY_PATH"] = f"{folder}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def find_system_tesseract() -> None:
    """Locate a Tesseract installed on the system (installer puts it in Program Files)."""
    import shutil
    if os.getenv("TESSERACT_CMD") or shutil.which("tesseract"):
        return
    candidates = []
    if sys.platform == "win32":
        for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            root = os.getenv(var)
            if root:
                candidates += [Path(root) / "Tesseract-OCR" / "tesseract.exe",
                               Path(root) / "Programs" / "Tesseract-OCR" / "tesseract.exe"]
    else:
        candidates += [Path(p) for p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract")]
    for path in candidates:
        if path.exists():
            os.environ["TESSERACT_CMD"] = str(path)
            return


def redirect_output_to_log() -> None:
    """With no console (pythonw / double-click), keep output in a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    base = Path(os.getenv("LOCALAPPDATA") or Path.home() / ".local" / "share") / "Modulo"
    base.mkdir(parents=True, exist_ok=True)
    log = open(base / "modulo.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log


def app_version() -> str:
    for folder in (BASE, BASE.parent):
        try:
            return (folder / "VERSION").read_text().strip()
        except OSError:
            continue
    return "dev"


def free_port() -> int:
    wanted = os.getenv("PORT")
    if wanted:
        return int(wanted)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve(app, port: int) -> None:
    try:
        from waitress import serve as waitress_serve
        waitress_serve(app, host="127.0.0.1", port=port, threads=4, ident="modulo")
    except ImportError:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


def wait_until_up(url: str, timeout: float = 20) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).read(1)
            return True
        except Exception:
            time.sleep(0.15)
    return False


def main() -> int:
    redirect_output_to_log()
    configure_bundled_tesseract()
    find_system_tesseract()
    from backend.app import create_app

    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    flask_app = create_app()
    app_state.flask = flask_app
    threading.Thread(target=serve, args=(flask_app, port), daemon=True).start()
    if not wait_until_up(url):
        print("Modulo could not start its local server.", file=sys.stderr)
        return 1

    if os.getenv("MODULO_NO_UI"):  # headless / smoke-test mode
        print(f"Modulo running at {url}", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0

    try:
        import webview  # native window; closing it exits the app
        webview.create_window(f"Modulo {app_version()}", url, width=1280, height=860, min_size=(900, 600))
        webview.start()
    except Exception as exc:
        print(f"Native window unavailable ({exc}); using the browser.", flush=True)
        webbrowser.open(url)
        print(f"Modulo is running at {url} - close the browser tab to quit.", flush=True)
        try:
            # The page pings /api/ping; quit once the tab has been closed for a while.
            started = time.time()
            while True:
                time.sleep(2)
                seen = getattr(app_state, "last_ping", 0)
                if seen and time.time() - seen > 45:
                    break
                if not seen and time.time() - started > 300:
                    break
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
