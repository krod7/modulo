# Modulo installer for Windows. Installs Python (if needed), all packages, Tesseract OCR,
# and creates a desktop icon. Run via "Install Modulo (Windows).bat".
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # makes Invoke-WebRequest much faster
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Source = Split-Path -Parent $PSScriptRoot
$Dest   = Join-Path $env:LOCALAPPDATA 'Modulo'
$AppDir = Join-Path $Dest 'app'
$Venv   = Join-Path $Dest 'venv'
$Temp   = Join-Path $env:TEMP 'modulo-setup'
$PythonVersion = '3.12.10'
New-Item -ItemType Directory -Force $Dest, $Temp | Out-Null
Start-Transcript -Path (Join-Path $Dest 'install.log') -Force | Out-Null

function Step($n, $text) { Write-Host ""; Write-Host "[$n/5] $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "      $text" -ForegroundColor Green }
function Warn($text) { Write-Host "      $text" -ForegroundColor Yellow }

# Run a native command, show its output, fail on non-zero exit code.
function Run([string]$Exe, [string[]]$Arguments) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0) { throw "Command failed ($code): $Exe $($Arguments -join ' ')" }
}
function Download([string]$Url, [string]$Path) {
    Write-Host "      Downloading $Url" -ForegroundColor DarkGray
    Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
}

function Find-Tesseract {
    $candidates = @(
        "$env:ProgramFiles\Tesseract-OCR\tesseract.exe",
        "${env:ProgramFiles(x86)}\Tesseract-OCR\tesseract.exe",
        "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $cmd = Get-Command tesseract -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Find-Python {
    # Prefer one we installed earlier, then the py launcher, then PATH. Needs 3.10-3.13.
    $own = Join-Path $Dest 'python\python.exe'
    if (Test-Path $own) { return $own }
    foreach ($v in '3.12', '3.11', '3.13', '3.10') {
        try {
            $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            $exe = & py "-$v" -c "import sys;print(sys.executable)" 2>$null
            $code = $LASTEXITCODE; $ErrorActionPreference = $old
            if ($code -eq 0 -and $exe -and (Test-Path $exe)) { return $exe }
        } catch { $ErrorActionPreference = 'Stop' }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike '*WindowsApps*') {
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $ver = & $cmd.Source -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        $ErrorActionPreference = $old
        if ($ver -in '3.10', '3.11', '3.12', '3.13') { return $cmd.Source }
    }
    return $null
}

try {
    Write-Host "=== Modulo installer ===" -ForegroundColor White
    if (-not (Test-Path (Join-Path $Source 'backend'))) {
        throw "Can't find the app files next to this installer. Extract the whole zip first, then run the installer again."
    }

    # 1 ---------------------------------------------------------------- app files
    Step 1 'Copying Modulo'
    New-Item -ItemType Directory -Force $AppDir | Out-Null
    foreach ($item in 'backend', 'frontend', 'assets', 'samples', 'docs') {
        $from = Join-Path $Source $item
        if (Test-Path $from) {
            $to = Join-Path $AppDir $item
            if (Test-Path $to) { Remove-Item $to -Recurse -Force }
            Copy-Item $from $to -Recurse -Force
        }
    }
    foreach ($file in 'modulo_launcher.py', 'requirements.txt', 'VERSION') {
        Copy-Item (Join-Path $Source $file) $AppDir -Force
    }
    $ver = (Get-Content (Join-Path $AppDir 'VERSION') -ErrorAction SilentlyContinue | Select-Object -First 1)
    Ok "Installed Modulo $ver to $AppDir"

    # 2 ---------------------------------------------------------------- Python
    Step 2 'Checking for Python'
    $py = Find-Python
    if ($py) {
        Ok "Using existing Python: $py"
    } else {
        Warn "Python not found - installing Python $PythonVersion for your user only (no admin needed)."
        $arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'amd64' }
        $installer = Join-Path $Temp "python-$PythonVersion-$arch.exe"
        Download "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-$arch.exe" $installer
        $target = Join-Path $Dest 'python'
        $p = Start-Process $installer -Wait -PassThru -ArgumentList @(
            '/quiet', 'InstallAllUsers=0', "TargetDir=$target", 'PrependPath=0', 'Include_launcher=0',
            'Include_test=0', 'Include_doc=0', 'Include_tcltk=0', 'Shortcuts=0', 'AssociateFiles=0')
        $py = Join-Path $target 'python.exe'
        if (-not (Test-Path $py)) { throw "Python install failed (exit code $($p.ExitCode))." }
        Ok "Installed Python $PythonVersion"
    }

    # 3 ---------------------------------------------------------------- packages
    Step 3 'Installing Modulo packages (this can take a few minutes)'
    $venvPy = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path $venvPy)) { Run $py @('-m', 'venv', $Venv) }
    Run $venvPy @('-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', 'pip')
    Run $venvPy @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', (Join-Path $AppDir 'requirements.txt'))
    Run $venvPy @('-m', 'pip', 'install', '--disable-pip-version-check', 'waitress>=3.0')   # production web server
    try {   # native app window; if this fails Modulo simply opens in the browser
        Run $venvPy @('-m', 'pip', 'install', '--disable-pip-version-check', 'pywebview>=5.0')
    } catch { Warn 'Native window component unavailable - Modulo will open in your browser instead.' }
    Run $venvPy @('-c', "import flask, waitress, cv2, pymupdf, pytesseract, networkx, numpy, PIL; print('Packages OK')")

    # 4 ---------------------------------------------------------------- Tesseract
    Step 4 'Checking for Tesseract OCR'
    $tess = Find-Tesseract
    if ($tess) {
        Ok "Found Tesseract: $tess"
    } else {
        Warn 'Tesseract not found - installing it (Windows will ask for permission).'
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            try {
                Run $winget.Source @('install', '-e', '--id', 'UB-Mannheim.TesseractOCR', '--silent',
                    '--accept-package-agreements', '--accept-source-agreements')
            } catch { Warn "winget install didn't finish: $($_.Exception.Message)" }
            $tess = Find-Tesseract
        }
        if (-not $tess) {
            $exe = Join-Path $Temp 'tesseract-setup.exe'
            Download 'https://github.com/UB-Mannheim/tesseract/releases/download/v5.5.0.20241111/tesseract-ocr-w64-setup-5.5.0.20241111.exe' $exe
            Start-Process $exe -ArgumentList '/S' -Verb RunAs -Wait
            $tess = Find-Tesseract
        }
        if ($tess) { Ok "Installed Tesseract: $tess" }
        else { Warn 'Could not install Tesseract automatically. Modulo will install, but text recognition will not work until Tesseract is installed (https://github.com/UB-Mannheim/tesseract/wiki).' }
    }

    # 5 ---------------------------------------------------------------- shortcuts
    Step 5 'Creating desktop icon'
    $icon = Join-Path $AppDir 'assets\modulo.ico'
    $shell = New-Object -ComObject WScript.Shell
    function New-Shortcut($path) {
        $s = $shell.CreateShortcut($path)
        $s.TargetPath       = Join-Path $Venv 'Scripts\pythonw.exe'
        $s.Arguments        = '"' + (Join-Path $AppDir 'modulo_launcher.py') + '"'
        $s.WorkingDirectory = $AppDir
        $s.IconLocation     = $icon
        $s.Description      = 'Modulo - flowchart to Python'
        $s.Save()
    }
    $desktop = [Environment]::GetFolderPath('Desktop')
    New-Shortcut (Join-Path $desktop 'Modulo.lnk')
    $startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'Modulo.lnk'
    New-Shortcut $startMenu
    Ok "Desktop icon created: $desktop\Modulo.lnk"

    Write-Host ""
    Write-Host "All done! Double-click the Modulo icon on your desktop to start." -ForegroundColor Green
    Stop-Transcript | Out-Null
    Start-Process (Join-Path $desktop 'Modulo.lnk')
}
catch {
    Write-Host ""
    Write-Host "Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "A log was saved to $Dest\install.log" -ForegroundColor Red
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
