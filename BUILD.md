# Building the standalone Modulo app

Run from source (no installer): `pip install -r requirements.txt && python modulo_launcher.py`

## Automatic (recommended)
Push a tag like `v1.0.0` (or run the "Build installers" workflow manually). GitHub Actions
builds `ModuloSetup.exe` (Windows), `Modulo.dmg` (macOS) and `Modulo-linux.tar.gz`, and attaches
them to a GitHub Release. Anyone can download from the Releases page.

## Manual
1. `pip install -r requirements-build.txt`
2. Copy Tesseract into `vendor/tesseract/` (binary + DLLs/dylibs + `tessdata/eng.traineddata`).
   Windows: copy `C:\Program Files\Tesseract-OCR` (UB Mannheim build).
3. `pyinstaller modulo.spec --noconfirm`  -> `dist/Modulo/`
4. Windows installer: `iscc installer\modulo.iss` -> `dist/ModuloSetup.exe`

## Before public release
- Code signing: Windows certificate (avoids SmartScreen warning); Apple Developer ID + notarization (macOS).
- PyMuPDF is AGPL-3.0: publish source / license accordingly, or switch to `pypdfium2` (README open decision #3).
- Add an app icon (`icon=` in `modulo.spec`, and `SetupIconFile` in `installer/modulo.iss`).
