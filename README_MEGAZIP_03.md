# MegaZIP 03 — Environment Sync and Embedded HTTP Probe

Esta entrega estabiliza el bootstrap embebido local antes de introducir CUDA o snapshots.

## Cambios

- `comfy-runtime env audit`: enumera los `requirements.txt` del núcleo y de cada custom node.
- `comfy-runtime env sync`: instala esos requisitos en el mismo entorno Python del engine.
- `comfy-runtime env sync --dry-run`: muestra lo que se instalaría.
- `comfy-runtime server-probe`: inicia el servidor HTTP embebido, comprueba el puerto y lo detiene.
- `doctor` muestra versión de PyTorch y disponibilidad/compilación CUDA.
- El bootstrap agrega explícitamente `--base-directory` y prepara/restaura `cwd`, `sys.argv` y `sys.path`.
- Limpieza más estricta del event loop al cerrar.
- `runtime-engine.toml` y los MegaZIP quedan excluidos de Git.
- `.gitattributes` fija LF para evitar las advertencias de CRLF.

## Flujo local

```powershell
python -m pip install -e ".[dev]"
pytest
comfy-runtime --config runtime-engine.toml env audit
comfy-runtime --config runtime-engine.toml env sync
comfy-runtime --config runtime-engine.toml doctor
comfy-runtime --config runtime-engine.toml probe
comfy-runtime --config runtime-engine.toml server-probe --hold-seconds 2
```

`env sync` no ejecuta scripts arbitrarios de los nodos; únicamente procesa archivos de requirements. Un requisito exclusivo de CUDA/Windows puede fallar sin impedir auditar los demás, salvo que se use `--strict`.

## Limpieza de archivos ya rastreados

El ZIP no elimina archivos locales. Para sacar del repositorio el TOML privado y MegaZIPs que ya fueron committeados, sin borrarlos del disco:

```powershell
git rm --cached --ignore-unmatch runtime-engine.toml
git rm --cached --ignore-unmatch MEGAZIP_01_ComfyUI_Runtime_Engine_Local_Foundation.zip
git rm --cached --ignore-unmatch MEGAZIP_02_ComfyUI_Runtime_Engine_Embedded_Bootstrap_REGENERADO.zip
```
