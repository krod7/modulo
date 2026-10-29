#!/usr/bin/env bash
# Modulo installer for macOS and Linux: installs Python + Tesseract if missing, sets up
# a private environment, and creates a desktop launcher.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then DEST="$HOME/Library/Application Support/Modulo"; else DEST="${XDG_DATA_HOME:-$HOME/.local/share}/Modulo"; fi
APP="$DEST/app"; VENV="$DEST/venv"

step() { printf '\n\033[36m[%s/5] %s\033[0m\n' "$1" "$2"; }
ok()   { printf '      \033[32m%s\033[0m\n' "$1"; }
warn() { printf '      \033[33m%s\033[0m\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
SUDO=""; if [ "$(id -u)" -ne 0 ] && have sudo; then SUDO="sudo"; fi

[ -d "$SRC/backend" ] || { echo "Can't find the app files next to this installer. Extract the whole zip first."; exit 1; }

py_ok() { "$1" -c 'import sys,venv; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)' 2>/dev/null; }
find_python() {
  for c in python3.12 python3.11 python3.13 python3.10 python3 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
    if have "$c" && py_ok "$c"; then command -v "$c"; return 0; fi
  done
  return 1
}

pkg_install() {  # $@ = generic names handled per distro below
  if have apt-get; then $SUDO apt-get update -y && $SUDO apt-get install -y "$@"
  elif have dnf; then $SUDO dnf install -y "$@"
  elif have pacman; then $SUDO pacman -S --noconfirm "$@"
  elif have zypper; then $SUDO zypper install -y "$@"
  else return 1; fi
}

ensure_brew() {
  if have brew; then return 0; fi
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && { eval "$($b shellenv)"; return 0; }; done
  warn "Installing Homebrew (needed to install Python and Tesseract). You may be asked for your password."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$($b shellenv)"; done
}

echo "=== Modulo installer ==="

step 1 "Copying Modulo"
mkdir -p "$APP"
for item in backend frontend assets samples docs; do
  [ -d "$SRC/$item" ] && { rm -rf "$APP/$item"; cp -R "$SRC/$item" "$APP/$item"; }
done
cp "$SRC/modulo_launcher.py" "$SRC/requirements.txt" "$SRC/VERSION" "$APP/"
ok "Installed to $APP"

step 2 "Checking for Python"
if PY="$(find_python)"; then ok "Using $PY"
else
  warn "Python 3.10-3.13 not found - installing."
  if [ "$OS" = "Darwin" ]; then ensure_brew; brew install python@3.12
  else pkg_install python3 python3-venv python3-pip || pkg_install python3 python3-pip || true; fi
  PY="$(find_python)" || { echo "Could not install Python automatically. Install Python 3.12 from python.org and re-run."; exit 1; }
fi
# Debian/Ubuntu ship venv separately
if ! "$PY" -c 'import ensurepip' 2>/dev/null; then pkg_install python3-venv || true; fi

step 3 "Installing Modulo packages (this can take a few minutes)"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip
"$VENV/bin/python" -m pip install --disable-pip-version-check -r "$APP/requirements.txt"
"$VENV/bin/python" -m pip install --disable-pip-version-check "waitress>=3.0"   # production web server
if [ "$OS" = "Darwin" ]; then   # native window; falls back to the browser if it fails
  "$VENV/bin/python" -m pip install --disable-pip-version-check "pywebview>=5.0" || warn "Native window unavailable - Modulo will open in your browser."
fi
"$VENV/bin/python" -c 'import flask, waitress, cv2, pymupdf, pytesseract, networkx, numpy, PIL; print("      Packages OK")'

step 4 "Checking for Tesseract OCR"
if have tesseract || [ -x /opt/homebrew/bin/tesseract ] || [ -x /usr/local/bin/tesseract ]; then ok "Tesseract already installed"
else
  warn "Tesseract not found - installing."
  if [ "$OS" = "Darwin" ]; then ensure_brew; brew install tesseract
  elif have apt-get; then pkg_install tesseract-ocr tesseract-ocr-eng
  elif have dnf; then pkg_install tesseract tesseract-langpack-eng
  else pkg_install tesseract tesseract-data-eng || pkg_install tesseract-ocr || true; fi
  if have tesseract || [ -x /opt/homebrew/bin/tesseract ]; then ok "Installed Tesseract"
  else warn "Could not install Tesseract automatically. Install it manually; text recognition needs it."; fi
fi

step 5 "Creating desktop icon"
RUN="$DEST/run-modulo.sh"
cat > "$RUN" <<RUNEOF
#!/usr/bin/env bash
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
cd "$APP"
exec "$VENV/bin/python" "$APP/modulo_launcher.py" >> "$DEST/modulo.log" 2>&1
RUNEOF
chmod +x "$RUN"

if [ "$OS" = "Darwin" ]; then
  BUNDLE="$HOME/Desktop/Modulo.app"
  rm -rf "$BUNDLE"; mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
  printf '#!/bin/bash\nexec "%s"\n' "$RUN" > "$BUNDLE/Contents/MacOS/Modulo"; chmod +x "$BUNDLE/Contents/MacOS/Modulo"
  cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>Modulo</string><key>CFBundleDisplayName</key><string>Modulo</string>
<key>CFBundleExecutable</key><string>Modulo</string><key>CFBundleIdentifier</key><string>com.modulo.launcher</string>
<key>CFBundleIconFile</key><string>modulo</string><key>CFBundlePackageType</key><string>APPL</string>
<key>LSUIElement</key><false/></dict></plist>
PLIST
  if have sips && have iconutil; then
    SET="$(mktemp -d)/modulo.iconset"; mkdir -p "$SET"
    for s in 16 32 64 128 256 512; do sips -z $s $s "$APP/assets/modulo.png" --out "$SET/icon_${s}x${s}.png" >/dev/null; done
    for s in 16 32 128 256; do cp "$SET/icon_$((s*2))x$((s*2)).png" "$SET/icon_${s}x${s}@2x.png"; done
    iconutil -c icns "$SET" -o "$BUNDLE/Contents/Resources/modulo.icns" || true
  fi
  ok "Desktop icon created: $BUNDLE"
  open "$BUNDLE" || true
else
  ENTRY="[Desktop Entry]
Type=Application
Name=Modulo
Comment=Flowchart to Python
Exec=\"$RUN\"
Icon=$APP/assets/modulo.png
Terminal=false
Categories=Development;Education;"
  mkdir -p "$HOME/.local/share/applications"
  printf '%s\n' "$ENTRY" > "$HOME/.local/share/applications/modulo.desktop"
  DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
  if [ -d "$DESK" ]; then
    printf '%s\n' "$ENTRY" > "$DESK/Modulo.desktop"; chmod +x "$DESK/Modulo.desktop"
    have gio && gio set "$DESK/Modulo.desktop" metadata::trusted true 2>/dev/null || true
  fi
  ok "Desktop icon + app menu entry created"
  nohup "$RUN" >/dev/null 2>&1 &
fi
echo; printf '\033[32mAll done! Use the Modulo icon on your desktop to start it.\033[0m\n'
