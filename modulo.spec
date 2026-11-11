# PyInstaller spec. Build with:  pyinstaller modulo.spec --noconfirm
# Optional: place a Tesseract folder (binary + DLLs + tessdata/) at ./vendor/tesseract
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

datas = [("frontend", "frontend"), ("assets", "assets")]
if Path("samples").exists():
    datas.append(("samples", "samples"))
if Path("accepted_flowcharts").exists():
    datas.append(("accepted_flowcharts", "accepted_flowcharts"))
if Path("vendor/tesseract").exists():
    datas.append(("vendor/tesseract", "tesseract"))

binaries, hidden = [], ["waitress"]
for pkg in ("pymupdf", "cv2"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h
try:
    d, b, h = collect_all("webview")
    datas += d; binaries += b; hidden += h
except Exception:
    pass

a = Analysis(["modulo_launcher.py"], pathex=["."], datas=datas, binaries=binaries,
             hiddenimports=hidden, excludes=["tkinter", "matplotlib", "pytest"])
pyz = PYZ(a.pure)
icon_path = str(Path("assets") / "modulo.ico") if sys.platform == "win32" else None
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Modulo",
          console=False, icon=icon_path)
coll = COLLECT(exe, a.binaries, a.datas, name="Modulo")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="Modulo.app", bundle_identifier="com.westtexassystemsolutions.modulo")
