HOTFIX RuntimeEngine08 — force configured snapshot residents fully onto CUDA
Base: RuntimeEngine07 behavior preserved (Python 3.10 tomli compatibility + subprocess diagnostics)

Production change:
- src/comfyui_runtime_engine/embedded/warmup.py
Preserved RuntimeEngine07 files are included so this incremental package is self-contained:
- src/comfyui_runtime_engine/config.py
- src/comfyui_runtime_engine/modal/adapter.py
- pyproject.toml

Problem proven by supplied Modal log:
Flux2TEModel_ (17180 MB) and Flux2 (33813 MB) were only staged/prepared for dynamic VRAM loading. SnapshotLifecycle then correctly blocked because configured residents were not fully loaded on CUDA.

Fix:
After the warmup prompt and before snapshot manifest validation, promote ONLY configured/tagged resident ModelPatchers with:
  model_management.load_models_gpu(patchers, force_full_load=True)
Then synchronize CUDA and run the existing strict snapshot readiness check.
No non-resident model is forced resident. Snapshot validation remains fail-closed.

Validation:
python -m py_compile src/comfyui_runtime_engine/embedded/warmup.py src/comfyui_runtime_engine/config.py src/comfyui_runtime_engine/modal/adapter.py
PYTHONPATH=src pytest -q tests/test_modal_adapter.py tests/test_modal_adapter_diagnostics.py tests/test_python310_tomli_compat.py tests/test_snapshot_residents_force_full_cuda.py
Result: 8 passed
