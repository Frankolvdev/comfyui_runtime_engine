# MegaZIP 08 — Resident Model Foundation

## Objetivos

1. Corregir `gpu-server-probe-repeat` para usar el Python exacto del `.venv`
   mediante `sys.executable`.
2. Crear el inventario y plan determinista de modelos residentes.
3. Resolver cada modelo por ruta relativa exacta o por nombre único.
4. Detectar modelos faltantes y nombres ambiguos.
5. Estimar el tamaño total de los pesos configurados.
6. Mantener la carga real fuera de esta etapa.

## Por qué todavía no se carga directamente el archivo

Un `.safetensors` puede representar un checkpoint, UNET/diffusion model, CLIP,
VAE, LoRA o un modelo de un custom node. El engine no debe adivinar qué loader
usar. La carga real se realizará con un workflow de warmup de ComfyUI, para
preservar compatibilidad con nodos estándar y personalizados.

## Configuración

En `runtime-engine.toml`:

```toml
[snapshot]
enabled = false
gpu_enabled = false
resident_models = [
  "diffusion_models/mi_modelo_pesado.safetensors",
  "text_encoders/mi_encoder.safetensors",
]

[residency]
model_roots = [
  # "D:/modelos/ComfyUI/models",
]
strict = true
execution_reserve_gb = 8.0
```

Si `model_roots` queda vacío, se usa:

```text
<comfyui_path>/models
```

## Pruebas

```powershell
python -m pip install -e ".[dev]"
pytest

comfy-runtime --config runtime-engine.toml models scan
comfy-runtime --config runtime-engine.toml models plan
comfy-runtime --config runtime-engine.toml models verify

comfy-runtime --config runtime-engine.toml gpu-server-probe-repeat `
  --iterations 3 `
  --hold-seconds 2
```

`models verify` fallará deliberadamente mientras no configures modelos o no
existan en las raíces indicadas.

## Próximo avance

El siguiente MegaZIP añadirá:

- workflow de warmup configurable;
- ejecución de loaders dentro de ComfyUI;
- inspección de `comfy.model_management.loaded_models`;
- telemetría de VRAM antes/después;
- validación de que los modelos configurados quedaron materializados.
