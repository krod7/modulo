; Inno Setup script -> ModuloSetup.exe. Run after PyInstaller: iscc installer\modulo.iss
[Setup]
AppName=Modulo
AppVersion=1.0.0
AppPublisher=West Texas System Solutions
DefaultDirName={autopf}\Modulo
DefaultGroupName=Modulo
OutputBaseFilename=ModuloSetup
OutputDir=..\dist
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
[Files]
Source: "..\dist\Modulo\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
[Icons]
Name: "{group}\Modulo"; Filename: "{app}\Modulo.exe"
Name: "{autodesktop}\Modulo"; Filename: "{app}\Modulo.exe"; Tasks: desktopicon
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"
[Run]
Filename: "{app}\Modulo.exe"; Description: "Launch Modulo"; Flags: postinstall nowait
