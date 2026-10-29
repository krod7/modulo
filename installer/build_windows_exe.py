"""Build dist/ModuloSetup.exe (Windows installer) from any OS.

Packs the app + PowerShell installer into a zip, embeds it in a tiny C stub and
cross-compiles with Zig (pip install ziglang). No Windows machine needed.

    pip install ziglang
    python installer/build_windows_exe.py
"""
from __future__ import annotations
import os, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
INCLUDE_DIRS = ("backend", "frontend", "assets", "samples", "docs")
INCLUDE_FILES = ("modulo_launcher.py", "requirements.txt", "VERSION", "installer/install-windows.ps1")


def build_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d in INCLUDE_DIRS:
            for dp, dn, fn in os.walk(ROOT / d):
                dn[:] = [x for x in dn if x != "__pycache__"]
                for f in fn:
                    if not f.endswith(".pyc"):
                        full = Path(dp) / f
                        z.write(full, full.relative_to(ROOT).as_posix())
        for f in INCLUDE_FILES:
            z.write(ROOT / f, f)
    return path.read_bytes()


def main() -> int:
    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = build_payload(tmp / "payload.zip")
        rows = (",".join(str(b) for b in data[i:i + 24]) for i in range(0, len(data), 24))
        (tmp / "payload.h").write_text(
            "static const unsigned char PAYLOAD[] = {\n" + ",\n".join(rows) +
            f"\n}};\nstatic const unsigned long PAYLOAD_LEN = {len(data)};\n")
        shutil.copy(ROOT / "assets" / "modulo.ico", tmp / "modulo.ico")   # rc files resolve paths relative to themselves
        (tmp / "res.rc").write_text('1 ICON "modulo.ico"\n')
        out = DIST / "ModuloSetup.exe"
        cmd = [sys.executable, "-m", "ziglang", "cc", "-target", "x86_64-windows-gnu", "-Os", "-s",
               f"-I{tmp}", str(ROOT / "installer" / "setup_stub.c"), str(tmp / "res.rc"), "-o", str(out)]
        subprocess.run(cmd, check=True)
    print(f"Built {out} ({out.stat().st_size // 1024} KB), version {(ROOT / 'VERSION').read_text().strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
