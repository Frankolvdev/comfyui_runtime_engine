import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from app.models.runtime_builder_config import RuntimeBuilderConfig


@dataclass
class ValidationIssue:
    level: str
    field: str
    message: str


class RuntimeBuilderService:
    """Validate and generate reproducible ComfyUI runtimes.

    This keeps the existing workflow/custom-node/model discovery flow intact.  The
    only dependency change is a conservative renderer that preserves PEP 508
    extras and markers instead of blindly inserting ``==``.
    """

    DEFAULT_MODAL_VOLUME_PATH = "/models"

    # Snapshot engine separado y versionable. Solo se usa en runtimes Modal.
    DEFAULT_RUNTIME_ENGINE_REPOSITORY = (
        "https://github.com/Frankolvdev/comfyui_runtime_engine.git"
    )
    DEFAULT_RUNTIME_ENGINE_REF = "main"
    DEFAULT_RUNTIME_ENGINE_INSTALL_PATH = "/opt/comfyui-runtime-engine"
    DEFAULT_MODAL_RESIDENT_MODELS = (
        "diffusion_models/realDream_klein9BV1.safetensors",
        "text_encoders/qwen_3_8b.safetensors",
    )


    PROTECTED_GPU_PACKAGES = {
        "torch", "torchvision", "torchaudio", "xformers", "triton",
        "onnxruntime-gpu", "flash-attn",
    }

    # Single source of truth used by the API endpoint, validation and exports.
    # MEGA31 accidentally removed this class attribute while the endpoint still
    # referenced it, causing GET /runtime-builder/config to fail at runtime.
    RECOMMENDED_PROFILE = {
        "id": "universal-modal-rtx5090",
        "label": "Universal GPU — Modal + RTX 5090",
        "python_version": "3.11",
        "cuda_version": "12.8.1",
        "pytorch_index_url": "https://download.pytorch.org/whl/cu128",
        "comfyui_version": "0.15.1",
        "comfyui_frontend_version": "1.39.19",
        "comfyui_commit": "3dd10a59c00248d00f0cb0ab794ff1bb9fb00a5f",
    }

    # Nombres alternativos de carpetas locales. El exportador siempre copia la
    # instalación local y nunca descarga ni reemplaza Custom Nodes desde Git.
    CUSTOM_NODE_ALIASES = {
        "comfyui-execute-python": ("execute-python",),
        "was-node-suite-comfyui": ("was-ns",),
        "comfyliterals": ("comfy-literals", "comfyui-comfyliterals"),
    }

    REQUIRED_CUSTOM_NODES = (
        {"name": "ComfyUI-Manager", "repository": "https://github.com/Comfy-Org/ComfyUI-Manager.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "rgthree-comfy", "repository": "https://github.com/rgthree/rgthree-comfy.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "ComfyUI-Easy-Use", "repository": "https://github.com/yolain/ComfyUI-Easy-Use.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "ComfyUI-Lora-Manager", "repository": "https://github.com/willmiao/ComfyUI-Lora-Manager.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "ComfyUI-KJNodes", "repository": "https://github.com/kijai/ComfyUI-KJNodes.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "ComfyUI_essentials", "repository": "https://github.com/cubiq/ComfyUI_essentials.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "was-node-suite-comfyui", "repository": "https://github.com/WASasquatch/was-node-suite-comfyui.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "ComfyUI-Logic", "repository": "https://github.com/theUpsider/ComfyUI-Logic.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "ComfyUI-Execute-Python", "repository": "https://github.com/mozhaa/ComfyUI-Execute-Python.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "ComfyLiterals", "repository": "https://github.com/M1kep/ComfyLiterals.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
        {"name": "Anomalous_Model_Browser", "repository": "https://github.com/DemonGatanjieu/Anomalous_Model_Browser.git", "commit": None, "enabled": True, "install_requirements": True, "required_by_default": True},
    )

    @staticmethod
    def sanitize_runtime_name(value: str | None) -> str:
        import unicodedata
        normalized = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
        normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
        normalized = re.sub(r"-+", "-", normalized)[:120].rstrip("-")
        if not normalized:
            normalized = "generation-runtime"
        if not normalized[0].isalpha():
            normalized = f"runtime-{normalized}"
        return normalized

    @staticmethod
    def merge_required_custom_nodes(nodes: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = [dict(item) for item in RuntimeBuilderService.REQUIRED_CUSTOM_NODES]
        keys = {str(item["repository"]).lower().removesuffix(".git") for item in merged}
        for item in nodes or []:
            repo = str(item.get("repository") or "").lower().removesuffix(".git")
            name = str(item.get("name") or "").lower()
            if repo in keys or any(str(x.get("name") or "").lower() == name for x in merged):
                continue
            merged.append(dict(item))
            if repo:
                keys.add(repo)
        return merged

    DEVELOPMENT_DEPENDENCIES = {
        "black", "flake8", "pytest", "pytest-cov", "pytest-asyncio",
        "coverage", "ruff", "isort", "mypy", "pre-commit", "tox",
        "nox", "bandit", "pylint", "autopep8", "yapf",
    }

    @staticmethod
    def _requirement_name(requirement: str) -> str:
        value = str(requirement or "").strip()
        value = value.split(";", 1)[0].strip()
        return re.split(r"\[|===|==|~=|!=|<=|>=|<|>|\s", value, maxsplit=1)[0].strip().lower()

    @staticmethod
    def is_runtime_dependency(requirement: str) -> bool:
        name = RuntimeBuilderService._requirement_name(requirement)
        return name not in RuntimeBuilderService.DEVELOPMENT_DEPENDENCIES and name not in RuntimeBuilderService.PROTECTED_GPU_PACKAGES

    @staticmethod
    def normalize_cuda_version(value: str | None) -> str:
        version = str(value or "").strip()
        if re.fullmatch(r"\d+\.\d+", version):
            return f"{version}.0"
        if re.fullmatch(r"\d+\.\d+\.\d+", version):
            return version
        raise ValueError(
            "La versión CUDA debe usar el formato mayor.menor o mayor.menor.parche, "
            "por ejemplo 12.8 o 12.8.0."
        )

    @staticmethod
    def _dependency_source(dependency: dict[str, Any]) -> str:
        return str(
            dependency.get("requirement")
            or dependency.get("package")
            or dependency.get("name")
            or ""
        ).strip()

    @staticmethod
    def render_requirement(dependency: dict[str, Any]) -> str:
        """Render one dependency as a valid PEP 508 requirement.

        Existing scanner output is accepted in either of these forms:
        - package=qrcode, version=[pil]       -> qrcode[pil]
        - package=onnxruntime-gpu, version="; marker" -> package; marker
        - package already containing extras/markers/specifier -> preserved
        - ordinary package + version -> package==version
        """

        raw_package = RuntimeBuilderService._dependency_source(dependency)
        raw_version = str(dependency.get("version") or "").strip()
        if not raw_package:
            raise ValueError("La dependencia no contiene package, name o requirement.")

        # mediapipe 0.10.0 no publica una distribución instalable para el
        # entorno Linux/Python usado por el runtime. 0.10.21 conserva la API
        # clásica utilizada por los Custom Nodes y sí dispone de wheels.
        package_name = raw_package.strip().lower()
        if package_name == "mediapipe" and raw_version in {"0.10.0", "==0.10.0"}:
            raw_version = "0.10.21"
        # PyAV 9.0.0 no ofrece wheel compatible con el runtime Linux/Python 3.10
        # y obliga a compilar contra FFmpeg, donde falla con toolchains actuales.
        # 12.3.0 mantiene la API usada por los nodos y dispone de wheels manylinux.
        if package_name == "av":
            version_number = raw_version.removeprefix("==").strip()
            if re.fullmatch(r"(?:8|9|10|11)(?:\.\d+){0,2}", version_number):
                raw_version = "12.3.0"

        # Some imported requirements already contain the whole PEP 508 string.
        if dependency.get("requirement"):
            candidate = raw_package
        elif raw_version.startswith("[") and raw_version.endswith("]"):
            # Fixes qrcode==[pil] generated by the previous renderer.
            candidate = f"{raw_package}{raw_version}"
        elif raw_version.startswith(";"):
            # Fixes onnxruntime-gpu==; marker.
            candidate = f"{raw_package}{raw_version}"
        elif not raw_version:
            candidate = raw_package
        elif re.match(r"^(===|==|~=|!=|<=|>=|<|>)", raw_version):
            candidate = f"{raw_package}{raw_version}"
        else:
            candidate = f"{raw_package}=={raw_version}"

        # Validación conservadora sin dependencias externas. El Runtime Builder
        # solo necesita impedir las formas que él mismo podía generar mal.
        if not candidate or candidate.startswith(("==", ";", "[")):
            raise ValueError(f"Dependencia Python inválida: {candidate!r}.")
        if "==[" in candidate or "==;" in candidate:
            raise ValueError(f"Dependencia Python inválida: {candidate!r}.")
        if any(char in candidate for char in ("\n", "\r", "\x00")):
            raise ValueError(f"Dependencia Python inválida: {candidate!r}.")
        return candidate

    @staticmethod
    def render_requirements(dependencies: list[dict[str, Any]]) -> list[str]:
        rendered: list[str] = []
        seen: set[str] = set()
        for dependency in dependencies:
            requirement = RuntimeBuilderService.render_requirement(dependency)
            if not RuntimeBuilderService.is_runtime_dependency(requirement):
                continue
            key = requirement.lower()
            if key not in seen:
                seen.add(key)
                rendered.append(requirement)
        return rendered

    @staticmethod
    def _is_modal(config: RuntimeBuilderConfig) -> bool:
        # Los perfiles nuevos declaran el proveedor explícitamente. Esta es la
        # fuente de verdad y permite múltiples runtimes Modal independientes.
        provider = str(getattr(config, "provider", "") or "").strip().lower()
        if provider:
            return provider == "modal"

        # Compatibilidad exclusiva con perfiles antiguos sin campo provider.
        values = [
            str(getattr(config, "target_platform", "") or ""),
            str(getattr(config, "notes", "") or ""),
        ]
        for item in getattr(config, "environment_variables", None) or []:
            values.extend([str(item.get("key") or ""), str(item.get("value") or "")])
        for volume in getattr(config, "volumes", None) or []:
            values.extend([
                str(volume.get("name") or ""),
                str(volume.get("mount_path") or volume.get("container_path") or volume.get("path") or ""),
            ])
        if any("modal" in value.lower() for value in values):
            return True

        # En el Runtime Builder actual, un Volume configurado junto con modelos
        # de estrategia volume representa el almacenamiento externo de Modal.
        # Esto evita depender de un campo de proveedor que aún no existe en UI.
        return bool(getattr(config, "volumes", None)) and RuntimeBuilderService._models_are_external(config)

    @staticmethod
    def _models_are_external(config: RuntimeBuilderConfig) -> bool:
        # Export Runtime nunca vuelve a resolver el workflow. Consume la lista
        # persistida del perfil, igual que el flujo Modal original funcional.
        enabled = [
            dict(item) for item in (config.models or [])
            if isinstance(item, dict) and item.get("enabled", True)
        ]
        if not enabled:
            return False
        external_strategies = {"volume", "external-volume", "external_volume", "mounted"}
        return all(str(item.get("strategy") or "").lower() in external_strategies for item in enabled)

    @staticmethod
    def _modal_volume_path(config: RuntimeBuilderConfig) -> str:
        for volume in config.volumes or []:
            provider = str(volume.get("provider") or volume.get("type") or "").lower()
            if provider == "modal" or str(volume.get("name") or "").lower().startswith("modal"):
                configured = str(
                    volume.get("container_path")
                    or volume.get("mount_path")
                    or volume.get("path")
                    or RuntimeBuilderService.DEFAULT_MODAL_VOLUME_PATH
                ).rstrip("/")
                if configured in {"/app/ComfyUI/models", "/models"}:
                    return RuntimeBuilderService.DEFAULT_MODAL_VOLUME_PATH
                return configured
        return RuntimeBuilderService.DEFAULT_MODAL_VOLUME_PATH

    @staticmethod
    def _extra_model_paths_yaml(base_path: str) -> str:
        return "\n".join(
            [
                "tryon_modal_volume:",
                f"  base_path: {base_path}",
                "  checkpoints: checkpoints",
                "  clip: text_encoders",
                "  clip_vision: clip_vision",
                "  configs: configs",
                "  controlnet: controlnet",
                "  diffusion_models: |",
                "    diffusion_models",
                "    unet",
                "  embeddings: embeddings",
                "  gligen: gligen",
                "  hypernetworks: hypernetworks",
                "  loras: loras",
                "  photomaker: photomaker",
                "  style_models: style_models",
                "  text_encoders: text_encoders",
                "  upscale_models: upscale_models",
                "  vae: vae",
                "  vae_approx: vae_approx",
                "  sam3: sam3",
                "",
            ]
        )

    @staticmethod
    def _tryon_runtime_guard_source() -> str:
        return 'import gc\nimport json\nimport os\nimport sys\nimport threading\nimport time\n\n_PATCH_LOCK = threading.Lock()\n_PATCHED = False\n_SAM3_CACHE = {}\n\n\ndef _enabled(name, default):\n    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}\n\n\ndef _log(event, **fields):\n    payload = {"event": event, "timestamp": time.time(), **fields}\n    print("[tryon-runtime-guard] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)\n\n\ndef _gpu_state():\n    try:\n        import torch\n        if not torch.cuda.is_available():\n            return {"cuda": False}\n        free, total = torch.cuda.mem_get_info()\n        return {\n            "cuda": True,\n            "free_gb": round(free / 1073741824, 3),\n            "total_gb": round(total / 1073741824, 3),\n            "allocated_gb": round(torch.cuda.memory_allocated() / 1073741824, 3),\n            "reserved_gb": round(torch.cuda.memory_reserved() / 1073741824, 3),\n        }\n    except Exception as exc:\n        return {"error": f"{type(exc).__name__}: {exc}"}\n\n\ndef _find_mapping_class(class_type):\n    for module in list(sys.modules.values()):\n        mapping = getattr(module, "NODE_CLASS_MAPPINGS", None)\n        if isinstance(mapping, dict) and class_type in mapping:\n            return mapping[class_type]\n    return None\n\n\ndef _patch_purge():\n    cls = _find_mapping_class("LayerUtility: PurgeVRAM V2")\n    if cls is None or getattr(cls, "_tryon_guarded", False):\n        return cls is not None\n    original = cls.purge_vram_v2\n\n    def guarded(self, anything, purge_cache, purge_models):\n        started = time.monotonic()\n        before = _gpu_state()\n        optimize = _enabled("TRYON_SELECTIVE_PURGE", "true")\n        fallback = _enabled("TRYON_FALLBACK_ORIGINAL_PURGE", "true")\n        threshold = float(os.getenv("TRYON_SELECTIVE_PURGE_MIN_FREE_GB", "28"))\n        try:\n            if not optimize or not purge_models:\n                result = original(self, anything, purge_cache, purge_models)\n                _log("purge_original", before=before, after=_gpu_state(), duration_s=round(time.monotonic()-started, 4))\n                return result\n\n            module = sys.modules.get(cls.__module__)\n            clear_memory = getattr(module, "clear_memory", None)\n            if callable(clear_memory):\n                clear_memory()\n            else:\n                gc.collect()\n            after_cache = _gpu_state()\n            free_gb = float(after_cache.get("free_gb", 0) or 0)\n            if free_gb >= threshold:\n                _log("purge_selective_keep_models", threshold_gb=threshold, before=before, after=after_cache, duration_s=round(time.monotonic()-started, 4))\n                return (anything,)\n\n            if _SAM3_CACHE:\n                _log("sam3_cache_released_for_pressure", entries=len(_SAM3_CACHE), free_gb=free_gb, threshold_gb=threshold)\n                _SAM3_CACHE.clear()\n                gc.collect()\n            result = original(self, anything, purge_cache, purge_models)\n            _log("purge_full_low_memory", threshold_gb=threshold, before=before, after=_gpu_state(), duration_s=round(time.monotonic()-started, 4))\n            return result\n        except BaseException as exc:\n            _log("purge_guard_error", error_type=type(exc).__name__, error=str(exc), fallback=fallback)\n            if fallback:\n                return original(self, anything, purge_cache, purge_models)\n            raise\n\n    cls.purge_vram_v2 = guarded\n    cls._tryon_guarded = True\n    _log("purge_patch_installed", class_module=cls.__module__)\n    return True\n\n\ndef _patch_sam3():\n    if not _enabled("TRYON_PROTECT_SAM3", "true"):\n        return True\n    cls = _find_mapping_class("TBGSAM3ModelLoaderAdvanced")\n    if cls is None or getattr(cls, "_tryon_guarded", False):\n        return cls is not None\n    function_name = getattr(cls, "FUNCTION", "")\n    original = getattr(cls, function_name, None)\n    if not function_name or not callable(original):\n        return False\n\n    def cached(self, *args, **kwargs):\n        key = (args, tuple(sorted((str(k), repr(v)) for k, v in kwargs.items())))\n        try:\n            hash(key)\n        except TypeError:\n            key = repr(key)\n        if key in _SAM3_CACHE:\n            _log("sam3_cache_hit", entries=len(_SAM3_CACHE), gpu=_gpu_state())\n            return _SAM3_CACHE[key]\n        started = time.monotonic()\n        result = original(self, *args, **kwargs)\n        _SAM3_CACHE[key] = result\n        _log("sam3_cache_store", entries=len(_SAM3_CACHE), duration_s=round(time.monotonic()-started, 4), gpu=_gpu_state())\n        return result\n\n    setattr(cls, function_name, cached)\n    cls._tryon_guarded = True\n    _log("sam3_patch_installed", class_module=cls.__module__, function=function_name)\n    return True\n\n\ndef _install():\n    global _PATCHED\n    deadline = time.monotonic() + 90\n    while time.monotonic() < deadline:\n        with _PATCH_LOCK:\n            purge_ok = _patch_purge()\n            sam_ok = _patch_sam3()\n            if purge_ok and sam_ok:\n                _PATCHED = True\n                _log("guard_ready", selective_purge=_enabled("TRYON_SELECTIVE_PURGE", "true"), protect_sam3=_enabled("TRYON_PROTECT_SAM3", "true"), fallback=_enabled("TRYON_FALLBACK_ORIGINAL_PURGE", "true"))\n                return\n        time.sleep(0.25)\n    _log("guard_partial_timeout", purge_found=_find_mapping_class("LayerUtility: PurgeVRAM V2") is not None, sam3_found=_find_mapping_class("TBGSAM3ModelLoaderAdvanced") is not None)\n\n\nthreading.Thread(target=_install, name="tryon-runtime-guard", daemon=True).start()\n\nNODE_CLASS_MAPPINGS = {}\nNODE_DISPLAY_NAME_MAPPINGS = {}\n'

    @staticmethod
    def _modal_runtime_engine_toml(volume_path: str) -> str:
        residents = "\n".join(
            f'  "{item}",'
            for item in RuntimeBuilderService.DEFAULT_MODAL_RESIDENT_MODELS
        )
        return f"""[runtime]
mode = "embedded"
comfyui_path = "/app/ComfyUI"
host = "127.0.0.1"
port = 8188
startup_timeout_seconds = 900
strict_version = true
supported_versions = ["0.15"]

[embedded]
allow_simulation_only = false
extra_args = ["--disable-api-nodes"]
shutdown_grace_seconds = 5.0
ensure_workspace = true

[snapshot]
enabled = true
gpu_enabled = true
resident_models = [
{residents}
]

[residency]
model_roots = ["{volume_path}"]
strict = true
execution_reserve_gb = 8.0
warmup_workflow = "/app/runtime/modal-snapshot-warmup.json"
warmup_timeout_seconds = 900
warmup_inputs = []

sam3_source_model = "{volume_path}/sam3/sam3.pt"
sam3_expected_model = "models/sam3/sam3.pt"
sam3_link_mode = "symlink"
sam3_replace_existing = false

[snapshot_lifecycle]
provider = "modal"
state_path = "/tmp/comfy-runtime-snapshot-state.json"
audit_path = "/tmp/comfy-runtime-snapshot-audit.json"

[diagnostics]
json_events = true
event_log = "/tmp/comfy-runtime-events.jsonl"
"""

    @staticmethod
    def _modal_snapshot_warmup_workflow() -> str:
        return json.dumps(
            {
                "runtime-resident-unet": {
                    "class_type": "UNETLoader",
                    "inputs": {
                        "unet_name": "realDream_klein9BV1.safetensors",
                        "weight_dtype": "default",
                    },
                },
                "runtime-resident-clip": {
                    "class_type": "CLIPLoader",
                    "inputs": {
                        "clip_name": "qwen_3_8b.safetensors",
                        "type": "flux2",
                        "device": "default",
                    },
                },
                "runtime-resident-unet-sink": {
                    "class_type": "TryonSnapshotWarmupSink",
                    "inputs": {
                        "value": ["runtime-resident-unet", 0],
                    },
                },
                "runtime-resident-clip-sink": {
                    "class_type": "TryonSnapshotWarmupSink",
                    "inputs": {
                        "value": ["runtime-resident-clip", 0],
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
        )

    @staticmethod
    def _modal_app(volume_name: str, volume_path: str, runtime_name: str) -> str:
        # Docker Desktop and RunPod keep using scripts/startup.sh unchanged.
        # Modal starts ComfyUI directly after snapshot restoration and exposes
        # it through a small ASGI reverse proxy so long-lived WebSockets remain
        # stable behind Modal's web ingress.
        return rf'''import asyncio
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal

APP_NAME = {json.dumps(runtime_name)}
VOLUME_NAME = {json.dumps(volume_name)}
VOLUME_PATH = {json.dumps(volume_path)}
COMFYUI_PORT = 8188
TRYON_RUNTIME_CONTRACT = "tryon.generation-runtime/v1"
STARTUP_TIMEOUT = int(os.getenv("TRYON_MODAL_STARTUP_TIMEOUT", "600"))

MODAL_GPU_ALIASES = {{"A10G": "A10"}}
MODAL_GPU_ALLOWED = {{
    "T4", "L4", "A10", "L40S", "A100", "A100-40GB", "A100-80GB",
    "RTX-PRO-6000", "H100", "H100!", "H200", "B200", "B200+", "B300",
}}


def _resolve_modal_gpu(value: str) -> str:
    requested = str(value or "L40S").strip()
    normalized = MODAL_GPU_ALIASES.get(requested.upper(), requested.upper())
    if normalized not in MODAL_GPU_ALLOWED:
        allowed = ", ".join(sorted(MODAL_GPU_ALLOWED))
        raise ValueError(
            f"GPU de Modal no válida: {{requested!r}}. Valores permitidos: {{allowed}}"
        )
    return normalized


GPU = _resolve_modal_gpu(os.getenv("TRYON_MODAL_GPU", "L40S"))
MIN_CONTAINERS = int(os.getenv("TRYON_MODAL_MIN_CONTAINERS", "0"))
MAX_CONTAINERS = int(os.getenv("TRYON_MODAL_MAX_CONTAINERS", "3"))
GENERATION_CONCURRENCY = int(os.getenv("TRYON_MODAL_CONCURRENCY", "1"))
INPUT_CONCURRENCY = int(os.getenv("TRYON_MODAL_INPUT_CONCURRENCY", "1000"))
SCALEDOWN_WINDOW = int(os.getenv("TRYON_MODAL_SCALEDOWN_WINDOW", "300"))
CPU_MEMORY_REQUEST_MB = int(os.getenv("TRYON_MODAL_CPU_MEMORY_REQUEST_MB", "32768"))
EXECUTION_TIMEOUT = int(os.getenv("TRYON_MODAL_EXECUTION_TIMEOUT", "1800"))
RUNTIME_ENGINE_ENABLED = os.getenv(
    "TRYON_MODAL_RUNTIME_ENGINE_ENABLED",
    "true",
).strip().lower() in {{"1", "true", "yes", "on"}}
RUNTIME_ENGINE_CONFIG = Path(
    os.getenv(
        "TRYON_MODAL_RUNTIME_ENGINE_CONFIG",
        "/app/runtime/runtime-engine.toml",
    )
)
RUNTIME_ENGINE_READY_PATH = Path(
    os.getenv(
        "TRYON_MODAL_RUNTIME_ENGINE_READY_PATH",
        "/tmp/comfy-runtime-modal-ready.json",
    )
)
RUNTIME_ENGINE_RESTORE_PATH = Path(
    os.getenv(
        "TRYON_MODAL_RUNTIME_ENGINE_RESTORE_PATH",
        "/tmp/comfy-runtime-modal-restore.json",
    )
)
RUNTIME_ENGINE_LOG_PATH = Path(
    os.getenv(
        "TRYON_MODAL_RUNTIME_ENGINE_LOG_PATH",
        "/tmp/comfy-runtime-modal.log",
    )
)

COMFYUI_ROOT = Path("/app/ComfyUI")
COMFYUI_MAIN = COMFYUI_ROOT / "main.py"
RUNTIME_ROOT = Path("/app/runtime")
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", VOLUME_PATH))
COMFY_USER_ROOT = Path(os.getenv("COMFY_USER_ROOT", "/tmp/comfyui-user"))
COMFY_DATABASE_URL = os.getenv(
    "COMFY_DATABASE_URL",
    f"sqlite:///{{COMFY_USER_ROOT / 'comfyui.db'}}",
)

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.from_dockerfile("Dockerfile.modal").pip_install("fastapi")


def _modal_trace(event: str, *, role: str, **fields) -> None:
    payload = {{
        "event": event,
        "role": role,
        "task_id": os.getenv("MODAL_TASK_ID"),
        "container_id": os.getenv("MODAL_CONTAINER_ID"),
        "function_call_id": os.getenv("MODAL_FUNCTION_CALL_ID"),
        "function_id": os.getenv("MODAL_FUNCTION_ID"),
        "region": os.getenv("MODAL_REGION"),
        "image_id": os.getenv("MODAL_IMAGE_ID"),
        "timestamp": time.time(),
        **fields,
    }}
    print("[tryon-modal-trace] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)


MODEL_DIAGNOSTICS_ENABLED = os.getenv("TRYON_MODAL_MODEL_DIAGNOSTICS", "true").strip().lower() in {{"1", "true", "yes", "on"}}
_MODEL_INPUT_HINTS = (
    "ckpt_name", "checkpoint", "model_name", "model_source", "unet_name",
    "vae_name", "clip_name", "control_net_name", "controlnet_name",
    "lora_name", "lora", "ipadapter_file", "pulid_file",
)


def _diagnostic_gpu_state() -> dict:
    state = {{}}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0:
            rows = []
            for line in completed.stdout.splitlines():
                values = [value.strip() for value in line.split(",")]
                if len(values) == 6:
                    rows.append({{
                        "index": values[0],
                        "name": values[1],
                        "memory_total_mb": values[2],
                        "memory_used_mb": values[3],
                        "memory_free_mb": values[4],
                        "utilization_percent": values[5],
                    }})
            state["gpus"] = rows
        elif completed.stderr.strip():
            state["error"] = completed.stderr.strip()[-500:]
    except Exception as exc:
        state["error"] = f"{{exc.__class__.__name__}}: {{exc}}"
    return state


def _looks_like_workflow(value) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    checked = 0
    matches = 0
    for node in value.values():
        if not isinstance(node, dict):
            continue
        checked += 1
        if isinstance(node.get("class_type"), str) and isinstance(node.get("inputs"), dict):
            matches += 1
        if checked >= 12:
            break
    return matches > 0 and matches == checked


def _decode_diagnostic_workflow(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def _find_payload_workflows(payload) -> list[dict]:
    workflows = []
    seen_ids = set()

    def visit(value, path: str) -> None:
        if isinstance(value, dict):
            if _looks_like_workflow(value):
                marker = id(value)
                if marker not in seen_ids:
                    seen_ids.add(marker)
                    workflows.append({{"path": path, "workflow": value}})
                return
            for key, child in value.items():
                if str(key).lower() == "workflow":
                    decoded = _decode_diagnostic_workflow(child)
                    if _looks_like_workflow(decoded):
                        marker = id(decoded)
                        if marker not in seen_ids:
                            seen_ids.add(marker)
                            workflows.append({{"path": f"{{path}}.{{key}}", "workflow": decoded}})
                        continue
                visit(child, f"{{path}}.{{key}}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{{path}}[{{index}}]")

    visit(payload, "payload")
    return workflows


def _workflow_model_inventory(workflow: dict) -> dict:
    loaders = []
    purge_nodes = []
    for raw_node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        node_id = str(raw_node_id)
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {{}}
        class_lower = class_type.lower()

        model_inputs = {{}}
        for key, value in inputs.items():
            key_lower = str(key).lower()
            if any(hint in key_lower for hint in _MODEL_INPUT_HINTS):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    model_inputs[str(key)] = value

        loader_like = bool(model_inputs) or any(
            token in class_lower
            for token in ("loader", "checkpoint", "controlnet", "ipadapter", "sam3")
        )
        if loader_like:
            loaders.append({{
                "node_id": node_id,
                "class_type": class_type,
                "model_inputs": model_inputs,
            }})

        if "purge" in class_lower or "unload" in class_lower or "empty cache" in class_lower:
            purge_nodes.append({{
                "node_id": node_id,
                "class_type": class_type,
                "purge_cache": inputs.get("purge_cache"),
                "purge_models": inputs.get("purge_models"),
            }})

    return {{
        "node_count": len(workflow),
        "loader_count": len(loaders),
        "purge_count": len(purge_nodes),
        "loaders": loaders,
        "purge_nodes": purge_nodes,
    }}


def _emit_model_diagnostics(payload, *, phase: str, execution_id: str) -> None:
    if not MODEL_DIAGNOSTICS_ENABLED:
        return
    try:
        workflows = _find_payload_workflows(payload)
        inventories = []
        for item in workflows:
            inventories.append({{
                "path": item["path"],
                **_workflow_model_inventory(item["workflow"]),
            }})
        _modal_trace(
            "model_diagnostics",
            role="pipeline_server",
            phase=phase,
            execution_id=execution_id,
            workflow_count=len(inventories),
            workflows=inventories,
            gpu=_diagnostic_gpu_state(),
        )
    except Exception as exc:
        _modal_trace(
            "model_diagnostics_error",
            role="pipeline_server",
            phase=phase,
            execution_id=execution_id,
            error_type=exc.__class__.__name__,
            error=str(exc),
        )


def _port_is_ready() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", COMFYUI_PORT), timeout=1):
            return True
    except OSError:
        return False


def _wait_until_ready(process: subprocess.Popen, timeout: int = STARTUP_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_is_ready():
            print(f"[modal] ComfyUI listo en el puerto {{COMFYUI_PORT}}.", flush=True)
            return
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"ComfyUI terminó antes de abrir el puerto {{COMFYUI_PORT}} "
                f"(código {{return_code}})."
            )
        time.sleep(1)
    raise TimeoutError(
        f"ComfyUI no abrió el puerto {{COMFYUI_PORT}} en {{timeout}} segundos."
    )


def _ensure_linux_machine_id() -> None:
    """Provide the machine-id expected by ComfyUI-Execute-Python on Modal."""
    primary = Path("/etc/machine-id")
    dbus = Path("/var/lib/dbus/machine-id")

    machine_id = ""
    if primary.is_file():
        try:
            candidate = primary.read_text(encoding="utf-8").strip().lower()
            if len(candidate) == 32 and all(char in "0123456789abcdef" for char in candidate):
                machine_id = candidate
        except OSError:
            pass
    if not machine_id:
        machine_id = uuid.uuid4().hex

    primary.parent.mkdir(parents=True, exist_ok=True)
    primary.write_text(machine_id + "\n", encoding="utf-8")
    dbus.parent.mkdir(parents=True, exist_ok=True)
    dbus.write_text(machine_id + "\n", encoding="utf-8")
    print(f"[runtime] Linux machine-id preparado para Execute Python: {{machine_id[:8]}}…", flush=True)


def _ensure_sam3_volume_link() -> None:
    """Expose the external SAM3 tree where TBG-SAM3 scans it directly."""
    source = MODELS_ROOT / "sam3"
    target = COMFYUI_ROOT / "models" / "sam3"

    if not source.is_dir():
        print(f"[runtime] SAM3 no enlazado: no existe el directorio {{source}}.", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                print(f"[runtime] SAM3 ya enlazado: {{target}} -> {{source}}", flush=True)
                return
        except OSError:
            pass
        target.unlink()
    elif target.exists():
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        else:
            raise RuntimeError(
                f"No se puede crear el enlace SAM3 porque {{target}} ya existe "
                "y contiene datos. No se eliminó ni sobrescribió nada."
            )

    target.symlink_to(source, target_is_directory=True)
    if not target.is_dir():
        raise RuntimeError(f"No se pudo crear el enlace SAM3: {{target}} -> {{source}}")
    print(f"[runtime] SAM3 enlazado desde el Volume: {{target}} -> {{source}}", flush=True)
    checkpoint = source / "sam3.pt"
    if not checkpoint.is_file():
        print(f"[runtime] Advertencia: no se encontró {{checkpoint}}.", flush=True)


def _prepare_runtime_directories() -> None:
    (COMFYUI_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (COMFY_USER_ROOT / "default" / "workflows").mkdir(parents=True, exist_ok=True)
    _ensure_linux_machine_id()
    _ensure_sam3_volume_link()
    print(f"[runtime] Modelos externos registrados desde: {{MODELS_ROOT}}", flush=True)
    print(f"[runtime] Directorio temporal de usuario: {{COMFY_USER_ROOT}}", flush=True)


def _run_performance_probe(env: dict[str, str]) -> None:
    probe = RUNTIME_ROOT / "scripts" / "performance_probe.py"
    if not probe.is_file():
        return
    try:
        subprocess.run([sys.executable, str(probe)], env=env, check=False)
    except OSError as exc:
        print(f"[modal] No se pudo ejecutar performance_probe.py: {{exc}}", flush=True)


def _proxy_app():
    from aiohttp import ClientSession, WSMsgType
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import Response

    web_app = FastAPI()
    upstream_http = f"http://127.0.0.1:{{COMFYUI_PORT}}"
    upstream_ws = f"ws://127.0.0.1:{{COMFYUI_PORT}}"
    hop_headers = {{
        "connection", "keep-alive", "proxy-authenticate",
        "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "content-length",
    }}

    def clean_headers(headers):
        return {{k: v for k, v in headers.items() if k.lower() not in hop_headers and k.lower() != "host"}}

    def upstream_request_headers(headers):
        forwarded = clean_headers(headers)
        if any(key.lower() == "origin" for key in forwarded):
            forwarded["Origin"] = upstream_http
        if any(key.lower() == "referer" for key in forwarded):
            forwarded["Referer"] = f"{{upstream_http}}/"
        return forwarded

    @web_app.post("/api/tryon/pipeline")
    async def execute_tryon_pipeline(request: Request):
        """Execute one complete workflow/Python pipeline in this GPU container."""
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="Pipeline payload must be a JSON object.",
            )

        if payload.get("runtime_contract") != "tryon.generation-runtime/v1":
            raise HTTPException(
                status_code=400,
                detail="Unsupported Generation Runtime contract.",
            )

        runtime_worker = RUNTIME_ROOT / "runpod_worker"
        if str(runtime_worker) not in sys.path:
            sys.path.insert(0, str(runtime_worker))

        try:
            from generation_runtime import GenerationRuntime

            runtime = GenerationRuntime(comfy_url=upstream_http)
            result = await asyncio.to_thread(runtime.execute, payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return result

    @web_app.api_route("/{{path:path}}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy_http(path: str, request: Request):
        _modal_trace(
            "proxy_http_request",
            role="web_proxy",
            method=request.method,
            path=f"/{{path}}",
            query=str(request.url.query or ""),
            user_agent=request.headers.get("user-agent"),
            origin=request.headers.get("origin"),
            referer=request.headers.get("referer"),
            forwarded_for=request.headers.get("x-forwarded-for"),
        )
        target = f"{{upstream_http}}/{{path}}"
        if request.url.query:
            target += f"?{{request.url.query}}"
        body = await request.body()
        async with ClientSession() as session:
            async with session.request(
                request.method,
                target,
                headers=upstream_request_headers(request.headers),
                data=body or None,
                allow_redirects=False,
            ) as upstream:
                content = await upstream.read()
                return Response(
                    content=content,
                    status_code=upstream.status,
                    headers=clean_headers(upstream.headers),
                    media_type=upstream.content_type if upstream.content_type else None,
                )

    @web_app.websocket("/{{path:path}}")
    async def proxy_websocket(websocket: WebSocket, path: str):
        headers = websocket.headers
        _modal_trace(
            "proxy_websocket_connect",
            role="web_proxy",
            path=f"/{{path}}",
            user_agent=headers.get("user-agent"),
            origin=headers.get("origin"),
            forwarded_for=headers.get("x-forwarded-for"),
        )
        target = f"{{upstream_ws}}/{{path}}"
        query = websocket.scope.get("query_string", b"").decode("latin-1")
        if query:
            target += f"?{{query}}"
        websocket_handshake_headers = hop_headers | {{
            "host", "sec-websocket-key", "sec-websocket-version",
            "sec-websocket-extensions", "sec-websocket-protocol",
        }}
        forwarded = {{
            key.decode("latin-1"): value.decode("latin-1")
            for key, value in websocket.scope.get("headers", [])
            if key.decode("latin-1").lower() not in websocket_handshake_headers
        }}
        forwarded["Origin"] = upstream_http
        async with ClientSession() as session:
            async with session.ws_connect(target, headers=forwarded, autoping=True) as upstream:
                await websocket.accept()

                async def client_to_upstream():
                    try:
                        while True:
                            message = await websocket.receive()
                            kind = message.get("type")
                            if kind == "websocket.disconnect":
                                break
                            if message.get("text") is not None:
                                await upstream.send_str(message["text"])
                            elif message.get("bytes") is not None:
                                await upstream.send_bytes(message["bytes"])
                    except WebSocketDisconnect:
                        pass
                    finally:
                        await upstream.close()

                async def upstream_to_client():
                    async for message in upstream:
                        if message.type == WSMsgType.TEXT:
                            await websocket.send_text(message.data)
                        elif message.type == WSMsgType.BINARY:
                            await websocket.send_bytes(message.data)
                        elif message.type in {{WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}}:
                            break

                tasks = [
                    asyncio.create_task(client_to_upstream()),
                    asyncio.create_task(upstream_to_client()),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
                _modal_trace(
                    "proxy_websocket_disconnect",
                    role="web_proxy",
                    path=f"/{{path}}",
                )

    return web_app


SNAPSHOT_MODEL_WARMUP_ENABLED = os.getenv(
    "TRYON_MODAL_SNAPSHOT_MODEL_WARMUP", "true"
).strip().lower() in {{"1", "true", "yes", "on"}}
SNAPSHOT_MODEL_WARMUP_TIMEOUT = int(
    os.getenv("TRYON_MODAL_SNAPSHOT_MODEL_WARMUP_TIMEOUT", "420")
)
SNAPSHOT_MODEL_WARMUP_TARGETS = (
    {{
        "node_id": "21578",
        "class_type": "VAELoader",
        "overrides": {{"vae_name": "flux2-vae.safetensors"}},
    }},
    {{
        "node_id": "21586",
        "class_type": "CLIPLoader",
        "overrides": {{
            "clip_name": "qwen_3_8b.safetensors",
            "type": "flux2",
            "device": "default",
        }},
    }},
    {{
        "node_id": "21584:21530",
        "class_type": "UNETLoader",
        "overrides": {{
            "unet_name": "Flux2-Klein-9B-True-v2-bf16.safetensors",
            "weight_dtype": "default",
        }},
    }},
    {{
        "node_id": "21584:21531",
        "class_type": "TBGSAM3ModelLoaderAdvanced",
        "overrides": {{
            "model_source": "sam3.pt",
            "device": "cuda",
        }},
    }},
)


def _write_snapshot_warmup_node() -> None:
    """Instala un nodo sumidero aislado usado solo durante la captura del snapshot."""
    try:
        custom_node_path = COMFYUI_ROOT / "custom_nodes" / "tryon_snapshot_warmup.py"
        custom_node_path.write_text(
            """import threading\n\n_WARM_OBJECTS = []\n_WARM_LOCK = threading.Lock()\n\nclass _AnyType(str):\n    def __ne__(self, other):\n        return False\n\n_ANY = _AnyType("*")\n\nclass TryonSnapshotWarmupSink:\n    @classmethod\n    def INPUT_TYPES(cls):\n        return {{"required": {{"value": (_ANY,)}}}}\n\n    RETURN_TYPES = ()\n    FUNCTION = "hold"\n    OUTPUT_NODE = True\n    CATEGORY = "tryon/internal"\n\n    def hold(self, value):\n        # Mantener una referencia fuerte hace que el objeto y sus pesos formen\n        # parte del snapshot. La promoción a GPU es best-effort y no bloquea\n        # el arranque si una clase concreta no expone un patcher compatible.\n        try:\n            import comfy.model_management as model_management\n            candidate = getattr(value, "patcher", None) or value\n            model_management.load_models_gpu([candidate])\n        except Exception as exc:\n            print(f"[tryon-warmup] GPU promotion skipped: {{exc.__class__.__name__}}: {{exc}}", flush=True)\n        with _WARM_LOCK:\n            _WARM_OBJECTS.append(value)\n        return ()\n\nNODE_CLASS_MAPPINGS = {{\n    "TryonSnapshotWarmupSink": TryonSnapshotWarmupSink,\n}}\nNODE_DISPLAY_NAME_MAPPINGS = {{\n    \"TryonSnapshotWarmupSink\": \"TryOn Snapshot Warmup Sink\",\n}}\n""",
            encoding="utf-8",
        )
    except Exception as exc:
        _modal_trace(
            "snapshot_model_warmup_node_error",
            role="pipeline_server",
            error_type=exc.__class__.__name__,
            error=str(exc),
        )


def _comfy_json_request(path: str, *, payload=None, timeout: int = 30):
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{{COMFYUI_PORT}}{{path}}"
    data = None
    headers = {{"Accept": "application/json"}}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode("utf-8")) if body else {{}}


def _default_loader_inputs(node_info: dict, overrides: dict) -> dict:
    required = ((node_info or {{}}).get("input") or {{}}).get("required") or {{}}
    result = {{}}
    for name, spec in required.items():
        if name in overrides:
            result[name] = overrides[name]
            continue
        if not isinstance(spec, list) or not spec:
            continue
        type_spec = spec[0]
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {{}}
        if "default" in options:
            result[name] = options["default"]
        elif isinstance(type_spec, list) and type_spec:
            result[name] = type_spec[0]
    # Solo agregar overrides que realmente existan en la definición del nodo.
    for name, value in overrides.items():
        if name in required:
            result[name] = value
    return result


def _run_snapshot_model_warmup() -> None:
    """Precarga fija y encapsulada; cualquier fallo conserva el snapshot normal."""
    if not SNAPSHOT_MODEL_WARMUP_ENABLED:
        _modal_trace("snapshot_model_warmup", role="pipeline_server", enabled=False)
        return

    started = time.monotonic()
    try:
        object_info = _comfy_json_request("/object_info", timeout=60)
        prompt = {{}}
        loaded_targets = []
        for index, target in enumerate(SNAPSHOT_MODEL_WARMUP_TARGETS):
            class_type = target["class_type"]
            info = object_info.get(class_type)
            if not isinstance(info, dict):
                raise RuntimeError(f"ComfyUI no registró el nodo {{class_type}}.")
            node_id = target["node_id"]
            prompt[node_id] = {{
                "class_type": class_type,
                "inputs": _default_loader_inputs(info, target["overrides"]),
            }}
            prompt[f"tryon-warmup-sink-{{index}}"] = {{
                "class_type": "TryonSnapshotWarmupSink",
                "inputs": {{"value": [node_id, 0]}},
            }}
            loaded_targets.append({{
                "node_id": node_id,
                "class_type": class_type,
                "inputs": prompt[node_id]["inputs"],
            }})

        client_id = f"tryon-snapshot-{{uuid.uuid4().hex}}"
        queued = _comfy_json_request(
            "/prompt",
            payload={{"prompt": prompt, "client_id": client_id}},
            timeout=60,
        )
        prompt_id = str(queued.get("prompt_id") or "")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI rechazó el warmup: {{queued}}")

        deadline = time.monotonic() + max(30, SNAPSHOT_MODEL_WARMUP_TIMEOUT)
        while time.monotonic() < deadline:
            history = _comfy_json_request(f"/history/{{prompt_id}}", timeout=30)
            record = history.get(prompt_id) if isinstance(history, dict) else None
            if isinstance(record, dict):
                status = record.get("status") if isinstance(record.get("status"), dict) else {{}}
                status_text = str(status.get("status_str") or "").lower()
                completed = bool(status.get("completed")) or status_text in {{"success", "completed"}}
                if completed:
                    _modal_trace(
                        "snapshot_model_warmup",
                        role="pipeline_server",
                        enabled=True,
                        completed=True,
                        duration_ms=int((time.monotonic() - started) * 1000),
                        prompt_id=prompt_id,
                        targets=loaded_targets,
                        gpu=_diagnostic_gpu_state(),
                    )
                    return
                messages = status.get("messages") if isinstance(status.get("messages"), list) else []
                if status_text in {{"error", "failed"}} or any(
                    isinstance(item, list) and item and item[0] == "execution_error"
                    for item in messages
                ):
                    raise RuntimeError(f"Warmup de ComfyUI falló: {{status}}")
            time.sleep(1)
        raise TimeoutError(
            f"Warmup no terminó en {{SNAPSHOT_MODEL_WARMUP_TIMEOUT}} segundos."
        )
    except BaseException as exc:
        _modal_trace(
            "snapshot_model_warmup",
            role="pipeline_server",
            enabled=True,
            completed=False,
            fallback="normal_snapshot",
            duration_ms=int((time.monotonic() - started) * 1000),
            error_type=exc.__class__.__name__,
            error=str(exc),
            gpu=_diagnostic_gpu_state(),
        )
        print(
            "[modal] Warmup de modelos omitido por seguridad; "
            "se continuará con el snapshot normal.",
            flush=True,
        )


@app.cls(
    image=image,
    gpu=GPU,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    volumes={{VOLUME_PATH: models_volume}},
    timeout=EXECUTION_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    memory=CPU_MEMORY_REQUEST_MB,
    enable_memory_snapshot=True,
    experimental_options={{"enable_gpu_snapshot": True}},
)
@modal.concurrent(max_inputs=GENERATION_CONCURRENCY)
class ComfyUIServer:
    def _start_process(self) -> None:
        if not COMFYUI_MAIN.is_file():
            raise RuntimeError(f"No se encontró ComfyUI en {{COMFYUI_MAIN}}.")

        env = os.environ.copy()
        env["RUNTIME_PROVIDER"] = "modal"
        env["COMFYUI_PORT"] = str(COMFYUI_PORT)
        env["MODELS_ROOT"] = str(MODELS_ROOT)
        env["COMFY_USER_ROOT"] = str(COMFY_USER_ROOT)
        env["COMFY_DATABASE_URL"] = COMFY_DATABASE_URL

        _prepare_runtime_directories()
        _write_snapshot_warmup_node()
        _run_performance_probe(env)

        extra_args = shlex.split(env.get("COMFYUI_EXTRA_ARGS", ""))
        command = [
            sys.executable,
            str(COMFYUI_MAIN),
            "--listen",
            "127.0.0.1",
            "--port",
            str(COMFYUI_PORT),
            "--user-directory",
            str(COMFY_USER_ROOT),
            "--database-url",
            COMFY_DATABASE_URL,
            *extra_args,
        ]
        print(f"[modal] Iniciando ComfyUI directamente: {{shlex.join(command)}}", flush=True)
        self.comfyui_process = subprocess.Popen(
            command,
            cwd=str(COMFYUI_ROOT),
            env=env,
            start_new_session=True,
        )
        _wait_until_ready(self.comfyui_process)

    @modal.enter(snap=True)
    def initialize_for_snapshot(self) -> None:
        _modal_trace(
            "container_snapshot_initialize",
            role="pipeline_server",
            snapshot_mode=(
                "runtime_engine_gpu_snapshot"
                if RUNTIME_ENGINE_ENABLED
                else "comfyui_gpu_warm_snapshot"
            ),
            comfyui_started=False,
            models_loaded=False,
            runtime_engine_enabled=RUNTIME_ENGINE_ENABLED,
        )
        os.environ["RUNTIME_PROVIDER"] = "modal"
        os.environ["COMFYUI_PORT"] = str(COMFYUI_PORT)

        if RUNTIME_ENGINE_ENABLED:
            # Import diferido: modal deploy importa este archivo primero en el host
            # local, donde el Engine no tiene por qué estar instalado. Este bloque
            # se ejecuta dentro de la imagen Modal, que sí instala el paquete.
            from comfyui_runtime_engine.modal import ModalSnapshotAdapter

            _prepare_runtime_directories()
            _write_snapshot_warmup_node()
            self.snapshot_adapter = ModalSnapshotAdapter(
                config_path=RUNTIME_ENGINE_CONFIG,
                host="127.0.0.1",
                port=COMFYUI_PORT,
                startup_timeout_seconds=STARTUP_TIMEOUT,
                ready_path=RUNTIME_ENGINE_READY_PATH,
                restore_path=RUNTIME_ENGINE_RESTORE_PATH,
                log_path=RUNTIME_ENGINE_LOG_PATH,
            )
            report = self.snapshot_adapter.prepare_snapshot()
            self.comfyui_process = self.snapshot_adapter.process
            _modal_trace(
                "container_snapshot_ready",
                role="pipeline_server",
                snapshot_mode="runtime_engine_gpu_snapshot",
                comfyui_started=True,
                models_loaded=True,
                runtime_engine_health=report,
                gpu=_diagnostic_gpu_state(),
            )
            print(
                "[modal] Runtime engine snapshot-ready; creando snapshot CPU+GPU.",
                flush=True,
            )
            return

        # Fallback exacto al comportamiento Modal anterior.
        print("[modal] Iniciando ComfyUI antes del snapshot de memoria.", flush=True)
        self._start_process()
        _run_snapshot_model_warmup()
        _modal_trace(
            "container_snapshot_ready",
            role="pipeline_server",
            snapshot_mode="comfyui_gpu_warm_snapshot",
            comfyui_started=True,
            models_loaded=True,
            runtime_engine_enabled=False,
            gpu=_diagnostic_gpu_state(),
        )
        print(
            "[modal] ComfyUI y modelos preparados; creando snapshot CPU+GPU.",
            flush=True,
        )

    @modal.enter(snap=False)
    def restore_after_snapshot(self) -> None:
        _modal_trace(
            "container_restore_start",
            role="pipeline_server",
            startup_mode=(
                "restored_runtime_engine_gpu_snapshot"
                if RUNTIME_ENGINE_ENABLED
                else "restored_comfyui_gpu_snapshot"
            ),
            runtime_engine_enabled=RUNTIME_ENGINE_ENABLED,
        )

        if RUNTIME_ENGINE_ENABLED:
            adapter = getattr(self, "snapshot_adapter", None)
            if adapter is None:
                raise RuntimeError(
                    "El snapshot no restauró ModalSnapshotAdapter."
                )
            report = adapter.after_restore()
            self.comfyui_process = adapter.process
            _modal_trace(
                "container_ready",
                role="pipeline_server",
                restored_from_snapshot=True,
                comfyui_snapshotted=True,
                models_snapshotted=True,
                startup_mode="restored_runtime_engine_gpu_snapshot",
                runtime_engine_health=report,
                gpu=_diagnostic_gpu_state(),
            )
            print(
                "[modal] Runtime engine restaurado y validado.",
                flush=True,
            )
            return

        # Fallback exacto al restore anterior.
        process = getattr(self, "comfyui_process", None)
        if process is None:
            raise RuntimeError(
                "El snapshot no restauró la referencia al proceso de ComfyUI."
            )

        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"ComfyUI no sobrevivió al restore del snapshot "
                f"(código {{return_code}})."
            )

        if _port_is_ready():
            print("[modal] ComfyUI restaurado desde snapshot y listo.", flush=True)
            _modal_trace(
                "container_ready",
                role="pipeline_server",
                restored_from_snapshot=True,
                comfyui_snapshotted=True,
                models_snapshotted=True,
                startup_mode="restored_comfyui_gpu_snapshot",
                gpu=_diagnostic_gpu_state(),
            )
            return

        print(
            "[modal] Snapshot restaurado; esperando que ComfyUI reactive el puerto.",
            flush=True,
        )
        _wait_until_ready(process)
        print("[modal] ComfyUI restaurado desde snapshot y listo.", flush=True)
        _modal_trace(
            "container_ready",
            role="pipeline_server",
            restored_from_snapshot=True,
            comfyui_snapshotted=True,
            models_snapshotted=True,
            startup_mode="restored_comfyui_gpu_snapshot",
            gpu=_diagnostic_gpu_state(),
        )

    @modal.method()
    def run_pipeline(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Pipeline payload must be a JSON object.")
        if payload.get("runtime_contract") != TRYON_RUNTIME_CONTRACT:
            raise ValueError("Unsupported Generation Runtime contract.")
        execution_id = str(payload.get("execution_id") or "")
        _emit_model_diagnostics(payload, phase="before_pipeline", execution_id=execution_id)
        _modal_trace("pipeline_start", role="pipeline_server", execution_id=execution_id)
        runtime_worker = RUNTIME_ROOT / "runpod_worker"
        if str(runtime_worker) not in sys.path:
            sys.path.insert(0, str(runtime_worker))
        from generation_runtime import GenerationRuntime
        runtime = GenerationRuntime(comfy_url=f"http://127.0.0.1:{{COMFYUI_PORT}}")
        started = time.monotonic()
        try:
            result = runtime.execute(payload)
        except BaseException as exc:
            _modal_trace(
                "pipeline_error",
                role="pipeline_server",
                execution_id=execution_id,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
            raise
        _modal_trace(
            "pipeline_end",
            role="pipeline_server",
            execution_id=execution_id,
            duration_ms=int((time.monotonic() - started) * 1000),
            status=result.get("status") if isinstance(result, dict) else None,
        )
        _emit_model_diagnostics(payload, phase="after_pipeline", execution_id=execution_id)
        return result

    @modal.asgi_app(requires_proxy_auth=True)
    def comfyui(self):
        return _proxy_app()

    @modal.exit()
    def shutdown(self) -> None:
        _modal_trace("container_exit", role="pipeline_server")
        process = getattr(self, "comfyui_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=15)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

'''

    @staticmethod
    def validate(
        config: RuntimeBuilderConfig,
        models_override: list[dict[str, Any]] | None = None,
    ) -> dict:
        issues: list[ValidationIssue] = []
        runtime_name = RuntimeBuilderService.sanitize_runtime_name(getattr(config, "runtime_name", None))
        if runtime_name != str(getattr(config, "runtime_name", "") or ""):
            issues.append(ValidationIssue("error", "runtime_name", f"El nombre debe usar formato Docker seguro. Sugerencia: {runtime_name}"))
        if not str(config.pytorch_index_url).rstrip("/").endswith("cu128"):
            issues.append(ValidationIssue("warning", "pytorch_index_url", "Para compatibilidad amplia con RTX 5090 y Modal se recomienda el índice cu128."))
        if RuntimeBuilderService.normalize_cuda_version(config.cuda_version) < "12.8.0":
            issues.append(ValidationIssue("warning", "cuda_version", "Para RTX 5090 se recomienda CUDA 12.8 o superior."))
        if not config.comfyui_commit:
            issues.append(
                ValidationIssue(
                    "warning",
                    "comfyui_commit",
                    "Conviene fijar un commit de ComfyUI para builds reproducibles.",
                )
            )
        if ":" not in config.registry_image:
            issues.append(
                ValidationIssue(
                    "warning",
                    "registry_image",
                    "La imagen no contiene un tag explícito; se agregará la versión del runtime.",
                )
            )
        if not re.match(r"^\d+\.\d+\.\d+([-.][A-Za-z0-9.]+)?$", config.runtime_version):
            issues.append(
                ValidationIssue(
                    "error",
                    "runtime_version",
                    "La versión debe seguir un formato semántico, por ejemplo 1.0.0.",
                )
            )

        names: set[str] = set()
        for index, node in enumerate(RuntimeBuilderService.merge_required_custom_nodes(config.custom_nodes)):
            name = str(node.get("name", "")).strip().lower()
            if not node.get("enabled", True):
                continue
            if name in names:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"custom_nodes.{index}.name",
                        "Existe un custom node duplicado.",
                    )
                )
            names.add(name)
            if not str(node.get("repository", "")).startswith(("https://", "git@")):
                issues.append(
                    ValidationIssue(
                        "error",
                        f"custom_nodes.{index}.repository",
                        "El repositorio del nodo no es válido.",
                    )
                )
            if not node.get("commit"):
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"custom_nodes.{index}.commit",
                        "El nodo no tiene commit fijo.",
                    )
                )

        if models_override is not None:
            enabled_models = [
                dict(model) for model in models_override
                if isinstance(model, dict) and model.get("enabled", True)
            ]
        else:
            enabled_models = [
                dict(model) for model in (config.models or [])
                if isinstance(model, dict) and model.get("enabled", True)
            ]
        for index, model in enumerate(enabled_models):
            if model.get("strategy") in {"image", "startup-download"} and not model.get("source_url"):
                issues.append(
                    ValidationIssue(
                        "error",
                        f"models.{index}.source_url",
                        "El modelo necesita una URL para esta estrategia.",
                    )
                )
            sha = model.get("sha256")
            if sha and not re.fullmatch(r"[a-fA-F0-9]{64}", sha):
                issues.append(
                    ValidationIssue(
                        "error",
                        f"models.{index}.sha256",
                        "SHA-256 debe contener 64 caracteres hexadecimales.",
                    )
                )

        for index, dependency in enumerate(config.python_dependencies or []):
            if not dependency.get("enabled", True):
                continue
            try:
                RuntimeBuilderService.render_requirement(dependency)
            except ValueError as exc:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"python_dependencies.{index}",
                        str(exc),
                    )
                )

        if RuntimeBuilderService._is_modal(config) and RuntimeBuilderService._models_are_external(config):
            if not config.volumes:
                issues.append(
                    ValidationIssue(
                        "error",
                        "volumes",
                        "Modal con modelos externos necesita un Volume configurado.",
                    )
                )

        return {
            "valid": not any(issue.level == "error" for issue in issues),
            "issues": [asdict(issue) for issue in issues],
            "summary": {
                "custom_nodes": len(
                    [n for n in RuntimeBuilderService.merge_required_custom_nodes(config.custom_nodes) if n.get("enabled", True)]
                ),
                "models": len(enabled_models),
                "python_dependencies": len(
                    [d for d in (config.python_dependencies or []) if d.get("enabled", True)]
                ),
                "volumes": len(config.volumes or []),
                "reproducible": bool(config.comfyui_commit)
                and all(
                    bool(n.get("commit"))
                    for n in RuntimeBuilderService.merge_required_custom_nodes(config.custom_nodes)
                    if n.get("enabled", True)
                ),
            },
        }

    @staticmethod
    def generate(
        config: RuntimeBuilderConfig,
        modal_volume_name: str | None = None,
    ) -> dict:
        # Todos los proveedores comparten el mismo método de exportación y
        # consumen exclusivamente los modelos persistidos en su propio perfil.
        # El análisis del workflow es una operación previa e independiente.
        modal_enabled = RuntimeBuilderService._is_modal(config)
        profile_models = [
            dict(model) for model in (config.models or [])
            if isinstance(model, dict) and model.get("enabled", True)
        ]
        validation = RuntimeBuilderService.validate(config, models_override=profile_models)
        if not validation["valid"]:
            errors = [item["message"] for item in validation["issues"] if item["level"] == "error"]
            raise ValueError("No se puede generar el runtime: " + " | ".join(errors))

        nodes = [n for n in RuntimeBuilderService.merge_required_custom_nodes(config.custom_nodes) if n.get("enabled", True)]
        deps = [d for d in (config.python_dependencies or []) if d.get("enabled", True)]
        models = profile_models

        node_lines: list[str] = []
        for node in nodes:
            folder = re.sub(r"[^A-Za-z0-9_.-]", "-", node["name"]).strip("-")
            node_lines.append(
                f"RUN git clone {node['repository']} /app/ComfyUI/custom_nodes/{folder}"
            )
            if node.get("commit"):
                node_lines.append(
                    f"RUN git -C /app/ComfyUI/custom_nodes/{folder} checkout {node['commit']}"
                )
            if node.get("install_requirements", True):
                node_lines.append(
                    "RUN if [ -f /app/ComfyUI/custom_nodes/"
                    f"{folder}/requirements.txt ]; then sed -Ei '/^(torch|torchvision|torchaudio|xformers|triton|onnxruntime-gpu|flash-attn)([<>=!~ ;]|$)/Id' "
                    f"/app/ComfyUI/custom_nodes/{folder}/requirements.txt && python -m pip install --no-cache-dir --constraint /tmp/runtime-constraints.txt -r "
                    f"/app/ComfyUI/custom_nodes/{folder}/requirements.txt; fi"
                )

        requirements = RuntimeBuilderService.render_requirements(deps)
        requirements_txt = "\n".join(requirements) + ("\n" if requirements else "")

        model_lines: list[str] = []
        for model in models:
            if model.get("strategy") == "image":
                model_lines.append(
                    "RUN mkdir -p $(dirname /app/ComfyUI/models/"
                    f"{model['target_path']}) && curl -fL '{model['source_url']}' "
                    f"-o /app/ComfyUI/models/{model['target_path']}"
                )

        external_models = RuntimeBuilderService._models_are_external(config)
        volume_path = RuntimeBuilderService._modal_volume_path(config)
        extra_paths_copy = ""
        if modal_enabled and external_models:
            extra_paths_copy = "COPY runtime-builder/extra_model_paths.yaml /app/ComfyUI/extra_model_paths.yaml"

        commit_line = (
            f"RUN git -C /app/ComfyUI checkout {config.comfyui_commit}"
            if config.comfyui_commit
            else ""
        )
        dockerfile = "\n".join(
            filter(
                None,
                [
                    f"FROM nvidia/cuda:{RuntimeBuilderService.normalize_cuda_version(config.cuda_version)}-cudnn-runtime-ubuntu22.04",
                    'ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PATH="/opt/conda/bin:$PATH" TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0"',
                    "RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git curl bzip2 ffmpeg libgl1 libopengl0 libglib2.0-0 build-essential pkg-config libgeos-dev libgdal-dev libcairo2-dev libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev && rm -rf /var/lib/apt/lists/*",
                    "RUN geos-config --version && ldconfig -p | grep -F libgeos_c.so",
                    "RUN curl -fL https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/Miniforge3-26.3.2-3-Linux-x86_64.sh -o /tmp/miniforge.sh && echo '848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c  /tmp/miniforge.sh' | sha256sum -c - && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh && conda install -y python=3.11 pip && conda clean -afy",
                    "RUN python --version && python -m pip install --upgrade 'pip>=25,<26' setuptools wheel",
                    f"RUN git clone {config.comfyui_repository} /app/ComfyUI",
                    commit_line,
                    (
                        "ARG COMFY_RUNTIME_ENGINE_GIT_URL="
                        + RuntimeBuilderService.DEFAULT_RUNTIME_ENGINE_REPOSITORY
                    ) if modal_enabled else "",
                    (
                        "ARG COMFY_RUNTIME_ENGINE_GIT_REF="
                        + RuntimeBuilderService.DEFAULT_RUNTIME_ENGINE_REF
                    ) if modal_enabled else "",
                    (
                        "RUN git clone --filter=blob:none "
                        "${COMFY_RUNTIME_ENGINE_GIT_URL} "
                        + RuntimeBuilderService.DEFAULT_RUNTIME_ENGINE_INSTALL_PATH
                        + " && git -C "
                        + RuntimeBuilderService.DEFAULT_RUNTIME_ENGINE_INSTALL_PATH
                        + " checkout ${COMFY_RUNTIME_ENGINE_GIT_REF}"
                    ) if modal_enabled else "",
                    (
                        "RUN python -m pip install --no-cache-dir "
                        + RuntimeBuilderService.DEFAULT_RUNTIME_ENGINE_INSTALL_PATH
                    ) if modal_enabled else "",
                    f"RUN python -m pip install --index-url {config.pytorch_index_url} torch torchvision torchaudio",
                    "RUN printf '%s\\n' 'transformers>=4.50.3,<5' > /tmp/runtime-constraints.txt && sed -Ei 's/^transformers.*$/transformers>=4.50.3,<5/I; /^(torch|torchvision|torchaudio|xformers|triton|onnxruntime-gpu|flash-attn)([<>=!~ ;]|$)/Id' /app/ComfyUI/requirements.txt && python -m pip install --constraint /tmp/runtime-constraints.txt -r /app/ComfyUI/requirements.txt",
                    *node_lines,
                    "COPY runtime-builder/requirements.txt /tmp/runtime-requirements.txt",
                    "RUN if [ -s /tmp/runtime-requirements.txt ]; then python -m pip install --constraint /tmp/runtime-constraints.txt -r /tmp/runtime-requirements.txt; fi",
                    'RUN set -eu; check_output="$(python -m pip check 2>&1)" && { printf \'%s\\n\' "$check_output"; exit 0; }; check_status=$?; printf \'%s\\n\' "$check_output"; unexpected="$(printf \'%s\\n\' "$check_output" | sed -E \'/^decord 0\\.6\\.0 is not supported on this platform$/d; /^[[:space:]]*$/d\')"; if [ -n "$unexpected" ]; then echo \'[runtime] pip check encontró errores no permitidos.\' >&2; exit "$check_status"; fi; echo \'[runtime] Advertencia conocida ignorada: decord 0.6.0 no declara soporte para esta plataforma.\'',
                    "RUN python -c 'import sys, torch, transformers; assert sys.version_info[:2] == (3, 11); assert torch.version.cuda and torch.version.cuda.startswith(\"12.8\"); assert int(transformers.__version__.split(\".\")[0]) < 5; print(sys.version); print(torch.__version__, torch.version.cuda); print(transformers.__version__)'",
                    *model_lines,
                    extra_paths_copy,
                    "COPY runtime-builder/tryon_runtime_guard/ /app/ComfyUI/custom_nodes/zzz_tryon_runtime_guard/",
                    "COPY runpod_worker /app/runtime/runpod_worker",
                    "WORKDIR /app/runtime/runpod_worker",
                    "RUN python -m pip install --constraint /tmp/runtime-constraints.txt -r requirements.txt",
                    "COPY runtime-builder/entrypoint.sh /app/runtime/entrypoint.sh",
                    "RUN chmod +x /app/runtime/entrypoint.sh",
                    'ENTRYPOINT ["/app/runtime/entrypoint.sh"]',
                ],
            )
        ) + "\n"

        comfy_args = "--listen 0.0.0.0 --port 8188" if modal_enabled else "--listen 127.0.0.1 --port 8188"
        health_port = 8188
        entrypoint = f"""#!/usr/bin/env bash
set -euo pipefail

# Modal ejecuta su propio runtime pasando el comando como argumentos al
# ENTRYPOINT de la imagen. Debemos cederle el control con exec "$@"; de lo
# contrario este script inicia ComfyUI por su cuenta y el lifecycle de
# modal_app.py nunca llega a ejecutarse. En Docker Desktop/RunPod normalmente
# no se pasan argumentos, por lo que se conserva el flujo tradicional.
if [ "$#" -gt 0 ]; then
  echo "[runtime] Delegando el proceso al runtime de Modal/Docker: $*"
  exec "$@"
fi

MODELS_ROOT="${{MODELS_ROOT:-/models}}"
WORKFLOWS_ROOT="${{WORKFLOWS_ROOT:-/workflows}}"
COMFY_USER_ROOT="${{COMFY_USER_ROOT:-$WORKFLOWS_ROOT}}"
mkdir -p /app/ComfyUI/models "$COMFY_USER_ROOT/default/workflows"
echo "[runtime] Modelos externos registrados desde: $MODELS_ROOT"
echo "[runtime] Workflows persistentes registrados en: $COMFY_USER_ROOT/default/workflows"
python - <<'PY_RUNTIME_PROBE' || true
import importlib.util, json, os
report = {{"provider": os.getenv("RUNTIME_PROVIDER", "docker")}}
try:
    import torch
    report.update({{"pytorch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}})
except Exception as exc:
    report["torch_error"] = str(exc)
for module in ("flash_attn", "xformers", "triton"):
    report[module] = importlib.util.find_spec(module) is not None
print("[runtime-performance] " + json.dumps(report, ensure_ascii=False, sort_keys=True))
PY_RUNTIME_PROBE
read -r -a EXTRA_ARGS <<< "${{COMFYUI_EXTRA_ARGS:-}}"
python /app/ComfyUI/main.py {comfy_args} --user-directory "$COMFY_USER_ROOT" "${{EXTRA_ARGS[@]}}" &
COMFY_PID=$!
for _ in $(seq 1 600); do
  curl -fsS http://127.0.0.1:{health_port}/system_stats >/dev/null && break
  sleep 1
done
if [ -f /app/runtime/runpod_worker/handler.py ] && [ "${{RUNTIME_PROVIDER:-}}" != "modal" ]; then
  python /app/runtime/runpod_worker/handler.py
fi
wait $COMFY_PID
"""

        runtime_manifest = {
            "contract": "runtime-builder/v2",
            "runtime_name": RuntimeBuilderService.sanitize_runtime_name(config.runtime_name),
            "gpu_profile": "universal-cu128",
            "gpu_targets": ["RTX 5090", "L4", "L40S", "A10G", "A100", "H100", "H200"],
            "name": config.name,
            "version": config.runtime_version,
            "platform": config.target_platform,
            "registry_image": config.registry_image,
            "comfyui": {
                "repository": config.comfyui_repository,
                "commit": config.comfyui_commit,
                "version": RuntimeBuilderService.RECOMMENDED_PROFILE["comfyui_version"],
                "frontend_version": RuntimeBuilderService.RECOMMENDED_PROFILE["comfyui_frontend_version"],
            },
            "compatibility_profile": RuntimeBuilderService.RECOMMENDED_PROFILE,
            "python": config.python_version,
            "cuda": config.cuda_version,
            "volumes": config.volumes or [],
            "provider": "modal" if modal_enabled else "docker",
            "model_storage": "external-volume" if external_models else "bundled",
        }
        custom_nodes_lock = {"nodes": nodes}
        models_manifest = {"models": models}
        env_example = (
            "\n".join(
                f"{item['key']}={'' if item.get('secret') else (item.get('value') or '')}"
                for item in (config.environment_variables or [])
            )
            + "\n"
        )

        result = {
            "dockerfile": dockerfile,
            "entrypoint": entrypoint,
            "requirements_txt": requirements_txt,
            "runtime_manifest": runtime_manifest,
            "custom_nodes_lock": custom_nodes_lock,
            "models_manifest": models_manifest,
            "env_example": env_example,
            "tryon_runtime_guard": RuntimeBuilderService._tryon_runtime_guard_source(),
        }

        if modal_enabled:
            result["modal_runtime_engine_toml"] = (
                RuntimeBuilderService._modal_runtime_engine_toml(volume_path)
            )
            result["modal_snapshot_warmup_workflow"] = (
                RuntimeBuilderService._modal_snapshot_warmup_workflow()
            )
            volume_name = str(modal_volume_name or "").strip()
            if not volume_name:
                volume_name = f"{RuntimeBuilderService.sanitize_runtime_name(config.runtime_name)}-models"
                for volume in config.volumes or []:
                    if volume.get("name"):
                        volume_name = str(volume["name"])
                        break
            result["modal_app"] = RuntimeBuilderService._modal_app(volume_name, volume_path, RuntimeBuilderService.sanitize_runtime_name(config.runtime_name))
            if external_models:
                result["extra_model_paths"] = RuntimeBuilderService._extra_model_paths_yaml(
                    volume_path
                )

        return result
