# MegaZIP 12 — Modal Snapshot Lifecycle Foundation

## Alcance

Este avance formaliza el contrato que utilizará Modal alrededor del GPU
snapshot, sin modificar Beam, RunPod, backend ni pipelines existentes.

Incluye:

- estado `snapshot-ready` generado dentro del mismo proceso de ComfyUI/CUDA;
- bloqueo si falta un modelo residente o no está completamente cargado;
- manifiesto atómico con identidad de cada modelo, bytes y VRAM;
- hook `before_snapshot()` para el futuro adaptador Modal;
- hook `after_restore()` para validar identidad y carga después de restaurar;
- simulación local del contrato;
- inspección del último manifiesto.

## Importante

La CLI local **no crea** un GPU Snapshot de Modal. Valida el mismo contrato y
produce el estado que el adaptador Modal deberá consumir dentro del proceso que
posee CUDA. Ejecutar el engine en un subprocess y snapshotear el padre no
preservaría la VRAM correcta.

## Configuración

Activa temporalmente:

```toml
[snapshot]
enabled = true
gpu_enabled = false
resident_models = [
  "diffusion_models/realDream_klein9BV1.safetensors",
]

[snapshot_lifecycle]
provider = "local-simulation"
state_path = ".runtime/snapshot-state.json"
```

`gpu_enabled` sigue en `false`: todavía no se conecta el hook a Modal.

## Pruebas

```powershell
python -m pip install -e ".[dev]"
pytest

comfy-runtime --config runtime-engine.toml snapshot simulate --hold-seconds 10
comfy-runtime --config runtime-engine.toml snapshot inspect
```

Debe aparecer:

```text
snapshot_state.snapshot_ready = true
snapshot_state.resident_count = 1
snapshot_state.residents[0].fully_loaded = true
snapshot_state.residents[0].device = cuda:0
success = true
```

## Próximo avance

El siguiente MegaZIP conectará este contrato con el runtime de Modal:

- ejecución del warmup durante inicialización;
- callback `before_snapshot` en el mismo proceso;
- validación `after_restore`;
- primer trabajo posterior a restauración;
- telemetría para detectar una recarga de pesos.
