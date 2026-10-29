# PyInstaller spec. Build with:  pyinstaller modulo.spec --noconfirm
# Optional: place a Tesseract folder (binary + DLLs + tessdata/) at ./vendor/tesseract
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

datas = [("frontend", "frontend")]
if Path("samples").exists():
    datas.append(("samples", "samples"))
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
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Modulo",
          console=False, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="Modulo")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="Modulo.app", bundle_identifier="com.westtexassystemsolutions.modulo")
