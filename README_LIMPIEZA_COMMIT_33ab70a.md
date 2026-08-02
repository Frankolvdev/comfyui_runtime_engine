# Limpieza del commit 33ab70a

Este paquete elimina únicamente archivos del backend que fueron copiados accidentalmente al repositorio `comfyui_runtime_engine`.

Se conservan:

- `pyproject.toml` versión 0.14.1
- `src/comfyui_runtime_engine/warmup/residency_guard.py`
- `tests/test_clip_residency_contract.py`
- `README_FIX_CLIP_RESIDENCY.md`

Archivos eliminados:

- `FIX_Runtime_Engine_CLIP_Resident_Detection.zip`
- `README_FIX_MODAL_SNAPSHOT_QWEN.md`
- `app/services/runtime_builder_service.py`
- `tests/test_megazip_14b_modal_engine_contract.py`
- `tests/test_megazip_14b_v3_contract.py`
