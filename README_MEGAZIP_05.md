# MegaZIP 05 — Dependency Repair + GPU Probe

Entrega incremental para `comfyui_runtime_engine`.

## Objetivos

1. Reparar la incompatibilidad real entre un `diffusers` antiguo y `huggingface_hub` moderno.
2. Garantizar que OpenCV incluya `cv2.ximgproc.guidedFilter` mediante una única variante contrib.
3. Añadir un diagnóstico CUDA local no destructivo antes de cargar modelos o habilitar snapshots.

## Nuevos comandos

```powershell
comfy-runtime --config runtime-engine.toml env repair --dry-run
comfy-runtime --config runtime-engine.toml env repair
comfy-runtime --config runtime-engine.toml env verify
comfy-runtime --config runtime-engine.toml gpu-probe
```

`gpu-probe` no inicia ComfyUI ni carga modelos. Solo comprueba PyTorch, compilación CUDA, disponibilidad real, dispositivos, `nvidia-smi` y una asignación mínima de un tensor.

## Alcance protegido

No implementa snapshots, warmup ni residencia de modelos. No modifica `comfyui-experimental`.
