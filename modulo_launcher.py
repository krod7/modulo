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


def set_windows_app_identity() -> None:
    """Give Windows a stable application identity so the taskbar groups Modulo correctly."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "WestTexasSystemSolutions.Modulo"
        )
    except Exception:
        pass


def apply_windows_window_icon(icon_path: Path) -> None:
    """Set both the title-bar and taskbar icons on the native pywebview window."""
    if sys.platform != "win32" or not icon_path.is_file():
        return
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        IMAGE_ICON, LR_LOADFROMFILE = 1, 0x0010
        WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
        SM_CXICON, SM_CYICON, SM_CXSMICON, SM_CYSMICON = 11, 12, 49, 50
        # pywebview creates its native window asynchronously after webview.start().
        for _ in range(100):
            hwnd = user32.FindWindowW(None, "Modulo")
            if hwnd:
                # Ask Windows for the ICO frame that matches each slot (DPI-aware) so the
                # title bar and taskbar never scale a single bitmap and look blurry.
                small = user32.LoadImageW(None, str(icon_path), IMAGE_ICON,
                                          user32.GetSystemMetrics(SM_CXSMICON),
                                          user32.GetSystemMetrics(SM_CYSMICON), LR_LOADFROMFILE)
                big = user32.LoadImageW(None, str(icon_path), IMAGE_ICON,
                                        user32.GetSystemMetrics(SM_CXICON),
                                        user32.GetSystemMetrics(SM_CYICON), LR_LOADFROMFILE)
                if small:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
                if big:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
                # Keep the HICONs alive for the lifetime of the process.
                globals()["_MODULO_HICONS"] = (small, big)
                return
            time.sleep(0.1)
    except Exception:
        pass


class WindowApi:
    """Exposed to the page as window.pywebview.api so it can toggle full screen.

    Underscore-prefixed attributes are not exposed to JavaScript by pywebview.
    """

    def __init__(self) -> None:
        self._window = None
        self._fullscreen = False

    def toggle_fullscreen(self) -> bool:
        """Switch between the maximized window and true full screen; returns the new state."""
        if self._window is None:
            return False
        self._window.toggle_fullscreen()
        self._fullscreen = not self._fullscreen
        return self._fullscreen


def main() -> int:
    set_windows_app_identity()
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
        icon_path = BASE / "assets" / "modulo.ico"
        if not icon_path.is_file():
            icon_path = Path(__file__).resolve().parent / "assets" / "modulo.ico"
        # pywebview's create_window() does not accept an icon= argument.
        # The executable icon is set by PyInstaller; apply_windows_window_icon()
        # sets the native title-bar/taskbar icons after the window is created.
        # Opens maximized (windowed full size); the page's Full screen button / F11 toggles true full screen.
        window_api = WindowApi()
        window_api._window = webview.create_window(
            "Modulo", url, width=1280, height=860, min_size=(900, 600),
            maximized=True, js_api=window_api,
        )
        if sys.platform == "win32":
            threading.Thread(target=apply_windows_window_icon, args=(icon_path,), daemon=True).start()
        start_kwargs = {}
        if sys.platform.startswith("linux"):
            # GTK/Qt windows take their icon from start(); use the scalable SVG so it stays sharp.
            svg_icon = BASE / "assets" / "modulo-m-icon.svg"
            if not svg_icon.is_file():
                svg_icon = Path(__file__).resolve().parent / "assets" / "modulo-m-icon.svg"
            if svg_icon.is_file():
                start_kwargs["icon"] = str(svg_icon)
        try:
            webview.start(**start_kwargs)
        except TypeError:  # older pywebview without the icon= argument
            webview.start()
    except Exception as exc:
        print(f"Native application window failed to start: {exc}", file=sys.stderr, flush=True)
        if sys.platform == "win32":
            # Do not silently open a browser: on Windows Modulo is intended to be a
            # desktop application with its own branded taskbar and title-bar icon.
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    None,
                    "Modulo could not open its application window.\n\n"
                    "Make sure Microsoft Edge WebView2 Runtime is installed, then run Modulo again.\n\n"
                    f"Details: {exc}",
                    "Modulo - Startup Error", 0x10,
                )
            except Exception:
                pass
            return 1
        # Browser fallback remains available on platforms without a supported native GUI.
        webbrowser.open(url)
        print(f"Modulo is running at {url} - close the browser tab to quit.", flush=True)
        try:
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
