# MegaZIP 11 — Selective Residency Guard + SAM3 Link

## Corrección de SAM3

El workflow no se modifica.

Se conserva exactamente:

```text
model_source = "sam3.pt"
```

Antes de importar ComfyUI, el engine prepara:

```text
<comfyui-experimental>/models/sam3/sam3.pt
```

como enlace hacia:

```text
C:/pinokio/api/comfy_3.git/app/models/sam3/sam3.pt
```

En Windows, `auto` intenta primero un symlink y después un hardlink. En Modal,
la misma configuración puede usar `symlink`.

Como la prueba anterior descargó una copia, usa una vez:

```toml
sam3_replace_existing = true
```

Después de crear el enlace puedes volver a `false`.

## Purges del workflow

`warmup verify` ahora reporta los nodos cuyo nombre o título contiene términos
como:

- purge
- unload
- free
- VRAM
- empty cache
- cleanup

El workflow no se reescribe y los nodos continúan ejecutándose.

## Protección selectiva

Durante la ejecución:

1. El loader estándar de diffusion models etiqueta el `ModelPatcher` con la
   ruta real del archivo.
2. La etiqueta se propaga a clones creados por LoRAs.
3. `free_memory()` conserva únicamente los `LoadedModel` asociados a los
   archivos declarados en `resident_models`.
4. Los demás modelos continúan usando el comportamiento normal de ComfyUI.

La protección se limita a las APIs estándar de `comfy.model_management`.
El reporte mostrará si un custom node hace una purga fuera de esas APIs.

## Prueba doble en el mismo proceso

```powershell
python -m pip install -e ".[dev]"
pytest

comfy-runtime --config runtime-engine.toml warmup verify

comfy-runtime --config runtime-engine.toml warmup probe `
  --iterations 2 `
  --hold-seconds 10
```

Revisar:

```text
workflow_modified = false
asset_links.sam3.status = created | already_linked
purge_nodes = [...]
residency_guard.protected_loaded_count >= 1
runs[0].history_status = success
runs[1].history_status = success
speedup_ratio > 1
```

Si el guard no identifica el modelo, el comando termina con `success = false`
aunque el workflow haya generado correctamente. Eso evita declarar residencia
sin evidencia.

## Fuera de alcance

- Modal GPU Snapshot;
- restauración del snapshot;
- integración con backend;
- protección de custom nodes que manipulen directamente
  `current_loaded_models` sin usar `free_memory`.
