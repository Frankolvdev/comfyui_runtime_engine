# HOTFIX RuntimeEngine05 — Python 3.10 compatibility metadata

Cambio único:
- `pyproject.toml`: `requires-python = ">=3.11"` -> `requires-python = ">=3.10"`

Motivo:
La imagen Modal del runtime usa Python 3.10.20 y pip rechazaba instalar
`comfyui-runtime-engine` antes de ejecutar cualquier código del Engine.

Este hotfix no modifica:
- backend
- Runtime Builder
- ComfyUI
- snapshot logic
- residency guard
- pipeline/workflows
- loaders/modelos
