# run_populate.ps1
# Script de PowerShell para ejecutar la población y limpieza de estudiantes

$apiBaseUrl = "https://apitesis.fanecorp.com"

# Allow selecting local or production
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "   SISTEMA DE ORIENTACION VOCACIONAL - POBLACION DE BD   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Seleccione el servidor objetivo:"
Write-Host "1) Produccion (https://apitesis.fanecorp.com)"
Write-Host "2) Local (http://localhost:8098)"
$choice = Read-Host "Elija una opcion (1 o 2, por defecto 1)"

if ($choice -eq "2") {
    $apiBaseUrl = "http://localhost:8098"
    Write-Host "Configurando objetivo: Local ($apiBaseUrl)" -ForegroundColor Yellow
} else {
    Write-Host "Configurando objetivo: Produccion ($apiBaseUrl)" -ForegroundColor Yellow
}

$env:API_BASE_URL = $apiBaseUrl
$env:PYTHONPATH = "."

Write-Host "Ejecutando script de poblacion..." -ForegroundColor Cyan
python scripts/populate_and_clean.py

Write-Host "Proceso terminado. Presione cualquier tecla para salir..." -ForegroundColor Gray
$null = [System.Console]::ReadKey()
