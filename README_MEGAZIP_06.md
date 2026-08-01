# MegaZIP 06 — CUDA Preparation + GPU Server Probe

## Alcance

Este avance corrige los problemas observados después del MegaZIP 05:

- fija `huggingface_hub>=0.34.0,<1.0`;
- mantiene `transformers>=4.57,<5`;
- mantiene `diffusers>=0.35.1,<0.36`;
- conserva una sola implementación real de `cv2` mediante
  `opencv-contrib-python==4.11.0.86`;
- trata como virtuales los avisos de `pip check` donde otros paquetes exigen
  por nombre `opencv-python` u `opencv-python-headless`, siempre que `cv2` y
  `guidedFilter` estén disponibles;
- recrea `temp` antes de verificar el workspace;
- añade preparación coordinada de PyTorch CUDA;
- añade un servidor embebido local sobre GPU, sin cargar modelos.

## Seguridad

`gpu setup --dry-run` no modifica el entorno.

El comando real reemplaza conjuntamente:

- torch 2.11.0
- torchvision 0.26.0
- torchaudio 2.11.0

usando por defecto el canal oficial CUDA 12.8.

## Orden de prueba

```powershell
python -m pip install -e ".[dev]"

comfy-runtime --config runtime-engine.toml env repair --dry-run
comfy-runtime --config runtime-engine.toml env repair
comfy-runtime --config runtime-engine.toml env verify

comfy-runtime --config runtime-engine.toml gpu setup --dry-run --channel cu128
comfy-runtime --config runtime-engine.toml gpu setup --channel cu128

comfy-runtime --config runtime-engine.toml gpu-probe
pytest
comfy-runtime --config runtime-engine.toml gpu-server-probe --hold-seconds 2
```

## Resultado esperado

`gpu-probe` debe mostrar:

```text
torch_version = 2.11.0+cu128
cuda_compiled = true
cuda_available = true
device_count >= 1
allocation_test = true
success = true
```

`gpu-server-probe` debe mostrar:

```text
same_process = true
server_bound = true
cpu_flag_removed = true
```

Todavía no se cargan modelos ni se crean snapshots.
