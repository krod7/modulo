"""Checks for the desktop launcher wiring (heartbeat route, version file)."""
import re
from pathlib import Path

from backend.app import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_heartbeat_route():
    assert create_app().test_client().post("/api/ping").status_code == 204


def test_version_file_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", (ROOT / "VERSION").read_text().strip())


def test_installer_payload_files_exist():
    for name in ("modulo_launcher.py", "requirements.txt", "installer/install-windows.ps1",
                 "installer/setup_stub.c", "assets/modulo.ico"):
        assert (ROOT / name).exists(), name
