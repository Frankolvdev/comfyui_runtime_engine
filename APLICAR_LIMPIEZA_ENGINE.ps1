$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$remove = @(
    "FIX_Runtime_Engine_CLIP_Resident_Detection.zip",
    "README_FIX_MODAL_SNAPSHOT_QWEN.md",
    "app/services/runtime_builder_service.py",
    "tests/test_megazip_14b_modal_engine_contract.py",
    "tests/test_megazip_14b_v3_contract.py"
)

foreach ($path in $remove) {
    if (Test-Path $path) {
        Remove-Item -Force -Recurse $path
        Write-Host "[limpieza] Eliminado: $path"
    }
}

if (Test-Path "app") {
    $remaining = Get-ChildItem "app" -Force -Recurse -ErrorAction SilentlyContinue
    if (-not $remaining) {
        Remove-Item -Force -Recurse "app"
        Write-Host "[limpieza] Eliminada carpeta app vacía."
    }
}

Write-Host "[ok] Limpieza terminada. El cambio CLIP permanece en src/comfyui_runtime_engine/warmup/residency_guard.py."
Write-Host "Ejecuta: `$env:PYTHONPATH = (Join-Path (Get-Location) 'src'); python -m pytest tests -v"
