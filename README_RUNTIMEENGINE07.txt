
RuntimeEngine07 - Python 3.10 TOML compatibility, preserving RuntimeEngine06 diagnostics

Base:
- comfyui_runtime_engine-main (2).zip
- RuntimeEngine06 diagnostics reapplied and preserved

Production files in this patch:
- src/comfyui_runtime_engine/config.py
- src/comfyui_runtime_engine/modal/adapter.py
- pyproject.toml

Fix:
- Python 3.11+ uses stdlib tomllib.
- Python 3.10 falls back to tomli as tomllib.
- pyproject installs tomli only when python_version < 3.11.
- RuntimeEngine06 subprocess diagnostics remain present.

No changes to:
- residency plan
- snapshot lifecycle
- warmup workflow
- ComfyUI bootstrap semantics
- CUDA/VRAM
- resident model selection
