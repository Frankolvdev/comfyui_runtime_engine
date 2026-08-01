# ComfyUI Runtime Engine — MegaZIP 02

Capa de ejecución seleccionable para ComfyUI normal o embebido.

## Modos disponibles

- `normal`: ejecuta el `main.py` original como subprocess.
- `embedded-simulation`: prueba lifecycle sin importar ComfyUI.
- `embedded probe`: inicializa ComfyUI real en el proceso del engine, sin abrir el puerto.
- `embedded`: inicia el servidor ComfyUI real en el proceso del engine.

ComfyUI 0.15.1 ya expone `start_comfyui(asyncio_loop)`. El engine consume esa
interfaz y conserva sin duplicación la carga de custom nodes, PromptServer,
PromptQueue, PromptExecutor, rutas y worker de prompts.

## Prueba local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item runtime-engine.example.toml runtime-engine.toml
```

Configura la ruta con `/` o comillas simples en TOML:

```toml
comfyui_path = 'C:\Users\frank\OneDrive\Documentos\comfyui-experimental'
```

Ejecuta:

```powershell
pytest
comfy-runtime --config runtime-engine.toml doctor
comfy-runtime --config runtime-engine.toml simulate
comfy-runtime --config runtime-engine.toml probe
```

El resultado correcto de `probe` debe incluir:

```json
{
  "same_process": true,
  "start_all_callable": true,
  "server_bound": false
}
```

Luego prueba el servidor embebido:

```powershell
comfy-runtime --config runtime-engine.toml start --mode embedded
```

La configuración recomendada para esta etapa usa `--cpu`, por lo que valida la
arquitectura embebida antes de introducir CUDA o snapshots.

## MegaZIP 03: entorno y prueba HTTP local

```powershell
comfy-runtime --config runtime-engine.toml env audit
comfy-runtime --config runtime-engine.toml env sync
comfy-runtime --config runtime-engine.toml server-probe --hold-seconds 2
```

Consulta `README_MEGAZIP_03.md` para el flujo completo y la limpieza de archivos locales ya rastreados.
