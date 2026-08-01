# MegaZIP 07 — Isolated GPU Probe + Resource Lifecycle

## Objetivo

Estabilizar el ciclo de vida local antes de cargar modelos.

## Cambios

- `gpu-probe` ahora ejecuta PyTorch/CUDA en un proceso auxiliar.
- El proceso que inicia ComfyUI ya no importa `torch` antes de `main.py`.
- `doctor` usa también el diagnóstico GPU aislado.
- Los probes cortos activan variables offline reversibles para reducir
  descargas y tareas remotas no esenciales.
- Se miden threads, tareas asyncio y la capacidad de crear `socketpair`.
- Se reportan threads nuevos después del cierre.
- Nuevo comando repetible que ejecuta cada prueba en un proceso nuevo:

```powershell
comfy-runtime --config runtime-engine.toml gpu-server-probe-repeat `
  --iterations 3 `
  --hold-seconds 2
```

ComfyUI no soporta importarse y reiniciarse varias veces dentro del mismo
intérprete. Por eso cada iteración repetida usa un proceso limpio.

## Pasos

```powershell
python -m pip install -e ".[dev]"
pytest

comfy-runtime --config runtime-engine.toml gpu-probe
comfy-runtime --config runtime-engine.toml doctor
comfy-runtime --config runtime-engine.toml gpu-server-probe --hold-seconds 2
comfy-runtime --config runtime-engine.toml gpu-server-probe-repeat --iterations 3 --hold-seconds 2
```

## Resultado esperado

En `gpu-server-probe`:

```text
torch_imported_in_parent_before_comfyui = false
same_process = true
server_bound = true
resources.socketpair_ok = true
```

En el probe repetido:

```text
iterations = 3
success = true
```

## Fuera de alcance

- carga de modelos;
- residencia en VRAM;
- warmup de workflow;
- snapshots CPU/GPU;
- integración con Modal.
