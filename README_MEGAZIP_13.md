# MegaZIP 13 — Snapshot Compatibility Audit

Último avance local antes de integrar Modal.

## Qué audita

La auditoría corre dentro del mismo proceso, después del warmup y después de
generar el estado `before_snapshot`.

Clasifica:

- intérprete y heap de Python;
- event loop de asyncio;
- PromptServer y sockets HTTP/WebSocket;
- PromptQueue;
- threads;
- tareas asyncio;
- subsistema de sockets;
- modelos CUDA residentes;
- rutas de modelos;
- estado del garbage collector.

## Clasificaciones

```text
preserve
recreate_after_restore
preserve_then_verify
```

Los sockets, listeners, event loop, tareas y threads se marcan para
reconstrucción o validación. Los modelos CUDA y sus identidades se conservan,
pero se verifican en `after_restore`.

## Configuración

Agrega:

```toml
[snapshot_lifecycle]
provider = "local-simulation"
state_path = ".runtime/snapshot-state.json"
audit_path = ".runtime/snapshot-audit.json"
```

## Ejecución

```powershell
python -m pip install --no-cache-dir -e ".[dev]"
pytest

comfy-runtime --config runtime-engine.toml snapshot simulate `
  --audit `
  --hold-seconds 10
```

## Resultado esperado

```text
snapshot_state.snapshot_ready = true
snapshot_audit.summary.snapshot_compatible = true
snapshot_audit.summary.blockers = []
```

Es normal que aparezcan componentes en:

```text
recreate_after_restore
```

No son fallos. Son el contrato para el primer MegaZIP de Modal.

## Inspección posterior

```powershell
comfy-runtime --config runtime-engine.toml snapshot audit-inspect
```

Archivo generado:

```text
.runtime/snapshot-audit.json
```

## Criterio para pasar a Modal

Podemos continuar cuando:

- no existan blockers;
- `cuda_resident_models` esté completamente cargado;
- `socket_subsystem.socketpair_ok` sea verdadero;
- el estado `snapshot_ready` sea verdadero;
- la auditoría produzca un plan explícito de `after_restore`.

La auditoría es conservadora y no promete que cualquier objeto arbitrario sea
snapshot-safe. Verifica el estado utilizado por este engine y separa los
recursos del sistema operativo que Modal deberá reconstruir.
