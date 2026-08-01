# MegaZIP 10 — Model Paths + Warmup Inputs + SAM3

## Correcciones

### 1. Model roots reales en ComfyUI

El engine registra cada `model_root` en `folder_paths` después de importar
ComfyUI y antes de validar el prompt.

Mapea:

- checkpoints
- diffusion_models y unet
- text_encoders y clip
- vae
- loras
- controlnet
- embeddings
- upscale_models
- sam3
- otros directorios estándar

Esto corrige las listas vacías de `UNETLoader`, `CLIPLoader` y `VAELoader`.

### 2. Dos imágenes del workflow

`warmup_inputs` copia las imágenes al directorio `input` real de ComfyUI con
el nombre exacto que el workflow API espera.

Ejemplo:

```toml
warmup_inputs = [
  { source = "C:/mis-imagenes/persona.png", target = "30d09464-0d1d-4baf-8d84-4f2fb3fd0cbb.png" },
  { source = "C:/mis-imagenes/prenda.jpg", target = "12f661a3367722ae211a2b282532e310.jpg" },
]
```

### 3. Solo modelos elegidos para residencia

Los demás modelos del workflow cargan normalmente. El engine solo reporta como
candidatos de residencia los incluidos en:

```toml
[snapshot]
resident_models = [
  "diffusion_models/realDream_klein9BV1.safetensors",
]
```

### 4. SAM3

Se registra:

```text
<model_root>/sam3
```

como categoría `sam3`.

Además, el workflow transforma automáticamente:

```text
model_source = "sam3.pt"
```

a:

```text
model_source = "local (auto-download)"
```

porque esos son los valores aceptados por `TBGSAM3ModelLoaderAdvanced`.

### 5. Sin timeout falso

Las llamadas HTTP ahora se ejecutan en un executor mientras el event loop del
servidor embebido sigue avanzando. Si ComfyUI rechaza el prompt, el engine
devuelve inmediatamente el JSON de validación en vez de terminar con `timed out`.

## Pruebas

```powershell
python -m pip install -e ".[dev]"
pytest

comfy-runtime --config runtime-engine.toml warmup verify
comfy-runtime --config runtime-engine.toml warmup probe --hold-seconds 10
```

## Resultado esperado

El reporte debe incluir:

```text
model_paths.available_counts.diffusion_models > 0
model_paths.available_counts.text_encoders > 0
model_paths.available_counts.vae > 0
model_paths.available_counts.sam3 >= 1
warmup_inputs.count = 2
prompt_changes.changes >= 1
history_status = success
```

La residencia o pinning real todavía no se activa en esta entrega.
