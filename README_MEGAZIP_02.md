# MegaZIP 02 — ComfyUI Embedded Bootstrap

Descomprimir encima del repositorio `comfyui_runtime_engine`, sin carpeta raíz adicional.

## Hito implementado

El engine puede importar ComfyUI 0.15.x y utilizar su función pública:

```python
main.start_comfyui(asyncio_loop)
```

ComfyUI devuelve su event loop, `PromptServer` y `start_all`. El engine no copia
`PromptServer`, `PromptExecutor`, rutas, cola, custom nodes ni el prompt worker.

## Prueba segura local

```powershell
pip install -e ".[dev]"
pytest
comfy-runtime --config runtime-engine.toml doctor
comfy-runtime --config runtime-engine.toml probe
```

`probe` inicializa ComfyUI real dentro del mismo PID, pero no enlaza el puerto HTTP.
Con `extra_args = ["--cpu", "--disable-api-nodes"]` evita CUDA en esta fase.

## Servidor embedded local

Cuando `probe` termine correctamente:

```powershell
comfy-runtime --config runtime-engine.toml start --mode embedded
```

Detener con `Ctrl+C`.

## Requisito de entorno

El proceso embedded debe usar el mismo entorno Python y dependencias de ComfyUI.
Si aparece una dependencia faltante, instala los requirements del ComfyUI
experimental en el `.venv` del engine o instala el engine editable dentro del
entorno de ComfyUI.

## No implementado aún

- snapshots CPU/GPU;
- warmup de modelos;
- residencia VRAM;
- integración Modal;
- cambios al backend o proveedores.
