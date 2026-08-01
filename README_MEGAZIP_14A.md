# MegaZIP 14A — Modal Snapshot Adapter

## Propósito

Primer avance productivo de la integración Modal. Este ZIP modifica únicamente
`comfyui_runtime_engine`.

No modifica:

- backend;
- Beam;
- RunPod;
- cancelaciones;
- pipeline de trabajos;
- despliegue Modal existente.

## Contrato entregado

```python
from comfyui_runtime_engine.modal import ModalSnapshotAdapter

adapter.prepare_snapshot()  # @modal.enter(snap=True)
adapter.after_restore()     # @modal.enter(snap=False)
adapter.health_check()
adapter.terminate()
```

El backend conservará los decoradores y la ejecución existente. Solo llamará a
este contrato.

## Arquitectura

El adaptador inicia un proceso hijo persistente:

```text
Modal class process
└── comfy-runtime modal serve
    └── ComfyUI embebido
        ├── custom nodes
        ├── workflow de warmup
        ├── resident guard
        ├── modelo residente CUDA
        └── servidor HTTP :8000
```

El proceso no termina después del warmup. Modal puede capturar el proceso padre,
el proceso hijo, la memoria Python y la memoria CUDA.

## Hooks

### `prepare_snapshot()`

- inicia `comfy-runtime modal serve`;
- espera el warmup;
- verifica todos los residentes;
- espera HTTP;
- falla si el estado es parcial;
- se usa desde `@modal.enter(snap=True)`.

### `after_restore()`

- no recarga modelos;
- comprueba que el proceso hijo siga vivo;
- comprueba HTTP;
- compara cantidad esperada de residentes;
- deja un manifiesto post-restore;
- se usa desde `@modal.enter(snap=False)`.

## Aplicación

Descomprimir encima del repositorio y ejecutar:

```powershell
python .\APLICAR_MEGAZIP_14A.py
python -m pip install --no-cache-dir -e ".[dev]"
pytest
```

## Prueba local disponible

La integración real requiere Modal. Localmente se validan:

- contrato JSON;
- fail-closed de health check;
- conteo exacto de residentes;
- CLI persistente;
- comportamiento anterior sin `persist_after_ready`.

## Configuración Modal

Usar como base:

```text
runtime-engine.modal.example.toml
```

La imagen debe contener:

```text
/app/ComfyUI
/app/runtime/runtime-engine.toml
/app/runtime/warmup-workflow-api.json
/app/runtime/warmup/body.jpg
/app/runtime/warmup/face.jpg
```

y montar:

```text
/models
```

## Uso esperado por el backend

```python
class Runtime:
    @modal.enter(snap=True)
    def prepare(self):
        self.snapshot_adapter = ModalSnapshotAdapter(
            config_path="/app/runtime/runtime-engine.toml",
            host="127.0.0.1",
            port=8000,
            startup_timeout_seconds=900,
        )
        self.snapshot_adapter.prepare_snapshot()

    @modal.enter(snap=False)
    def restore(self):
        self.snapshot_adapter.after_restore()

    @modal.web_server(8000, startup_timeout=900)
    def serve(self):
        pass
```

El MegaZIP 14B generará esta envoltura desde el backend sin sustituir el
pipeline de jobs ni las cancelaciones.

## Git

```powershell
git add .
git commit -m "feat: add persistent Modal GPU snapshot adapter"
git push
```
