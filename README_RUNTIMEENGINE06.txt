RuntimeEngine06 - Modal subprocess diagnostics, blindado

Base exacta: comfyui_runtime_engine-main (2).zip

Produccion modificada:
- src/comfyui_runtime_engine/modal/adapter.py

Prueba agregada:
- tests/test_modal_adapter_diagnostics.py

Cambios funcionales limitados a observabilidad/fail-fast:
- Mantiene el subprocess y el archivo canonico /tmp/comfy-runtime-modal.log.
- Fuerza PYTHONUNBUFFERED=1 solo para visibilidad del child process.
- Retransmite al stdout de Modal las lineas nuevas del log mientras espera snapshot_ready.
- Detecta si el subprocess muere antes de snapshot_ready y falla inmediatamente.
- En timeout incluye las ultimas lineas del log (max 12000 caracteres).

NO modifica:
- ComfyUI bootstrap
- residency plan / resident models
- warmup workflow
- snapshot lifecycle
- CUDA / VRAM
- model roots (/models)
- backend / pipeline / workflows

Validacion:
PYTHONPATH=src pytest -q tests/test_modal_adapter.py tests/test_modal_adapter_diagnostics.py
Resultado: 4 passed
python -m py_compile: OK
