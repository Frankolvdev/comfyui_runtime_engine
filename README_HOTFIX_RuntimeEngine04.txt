RuntimeEngine04 — compatibilidad ComfyUI 0.31 / snapshot Flux2 sin SAM3 obligatorio

Cambios:
- free_memory guard forwards pins_required y futuros kwargs de ComfyUI 0.31.
- Evita mutable default keep_loaded.
- SAM3 deja de ser requisito del lifecycle Modal si no está seleccionado como residente.
- No cambia políticas de residencia ni snapshot; solo compatibilidad/lifecycle.

Aplicar estos archivos al repo https://github.com/Frankolvdev/comfyui_runtime_engine.git (branch usado por backend: main) y subirlos antes de reconstruir Modal.
