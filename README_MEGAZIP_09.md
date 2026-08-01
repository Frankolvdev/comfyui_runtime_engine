# MegaZIP 09 — Workflow Warmup + VRAM Inspection

## Qué hace

- valida un workflow exportado en formato API;
- inicia ComfyUI embebido sobre la GPU local;
- envía el workflow a `/prompt`;
- espera su resultado en `/history/{prompt_id}`;
- mide VRAM antes y después;
- inspecciona `comfy.model_management.current_loaded_models`;
- mantiene el proceso vivo algunos segundos para validar residencia.

## Preparar el workflow

En ComfyUI activa el modo desarrollador y usa:

```text
Save (API Format)
```

Crea un workflow mínimo que cargue el modelo deseado y tenga una ruta ejecutable.
No uses el JSON normal de la interfaz; debe contener nodos con `class_type`.

Guárdalo, por ejemplo, como:

```text
F:\PROYECTOS PERSONALES\TRYON\comfyui_runtime_engine\warmup-workflow-api.json
```

Configura:

```toml
[residency]
warmup_workflow = "warmup-workflow-api.json"
warmup_timeout_seconds = 600
```

## Pruebas

```powershell
python -m pip install -e ".[dev]"
pytest

comfy-runtime --config runtime-engine.toml warmup verify
comfy-runtime --config runtime-engine.toml warmup probe --hold-seconds 10
```

El resultado debe mostrar:

```text
history_status = success
loaded_model_count_delta >= 1
vram_delta_reserved_bytes > 0
```

Dependiendo del loader, cargar un modelo puede consumir RAM primero y materializar
la VRAM únicamente cuando un nodo posterior lo usa. Por eso el workflow de warmup
debe ejecutar una operación mínima que fuerce su uso, no dejar el loader aislado.

## Aún fuera de alcance

- impedir unload automático;
- política de pinning;
- snapshot de Modal;
- restauración del snapshot.
