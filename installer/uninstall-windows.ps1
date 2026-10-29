$Dest = Join-Path $env:LOCALAPPDATA 'Modulo'
Remove-Item (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Modulo.lnk') -ErrorAction SilentlyContinue
Remove-Item (Join-Path ([Environment]::GetFolderPath('Programs')) 'Modulo.lnk') -ErrorAction SilentlyContinue
if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
Write-Host "Modulo has been removed. (Tesseract OCR was left in place; uninstall it from Settings > Apps if you want.)"
