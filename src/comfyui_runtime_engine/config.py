from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import tomllib

from .errors import ConfigurationError

_ALLOWED_MODES = {"normal", "embedded", "embedded-simulation"}


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    mode: str
    comfyui_path: Path
    python_executable: str = sys.executable
    host: str = "127.0.0.1"
    port: int = 8188
    startup_timeout_seconds: int = 120
    strict_version: bool = True
    supported_versions: tuple[str, ...] = ("0.15",)
    normal_extra_args: tuple[str, ...] = ()
    allow_simulation_only: bool = False
    embedded_extra_args: tuple[str, ...] = ()
    snapshot_enabled: bool = False
    gpu_snapshot_enabled: bool = False
    resident_models: tuple[str, ...] = ()
    json_events: bool = True
    event_log: Path = field(default_factory=lambda: Path(".runtime/events.jsonl"))

    @classmethod
    def from_toml(cls, path: str | Path) -> "RuntimeConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

        runtime = raw.get("runtime", {})
        normal = raw.get("normal", {})
        embedded = raw.get("embedded", {})
        snapshot = raw.get("snapshot", {})
        diagnostics = raw.get("diagnostics", {})

        mode = str(runtime.get("mode", "embedded-simulation")).strip().lower()
        if mode not in _ALLOWED_MODES:
            raise ConfigurationError(
                f"Invalid runtime.mode '{mode}'. Allowed: {sorted(_ALLOWED_MODES)}"
            )

        raw_comfyui_path = runtime.get("comfyui_path")
        if not raw_comfyui_path:
            raise ConfigurationError("runtime.comfyui_path is required")
        comfyui_path = Path(str(raw_comfyui_path)).expanduser()
        if not comfyui_path.is_absolute():
            comfyui_path = (config_path.parent / comfyui_path).resolve()
        else:
            comfyui_path = comfyui_path.resolve()

        event_log = Path(str(diagnostics.get("event_log", ".runtime/events.jsonl"))).expanduser()
        if not event_log.is_absolute():
            event_log = (config_path.parent / event_log).resolve()

        port = int(runtime.get("port", 8188))
        if port < 1 or port > 65535:
            raise ConfigurationError("runtime.port must be between 1 and 65535")

        return cls(
            mode=mode,
            comfyui_path=comfyui_path,
            python_executable=str(runtime.get("python_executable", sys.executable)),
            host=str(runtime.get("host", "127.0.0.1")),
            port=port,
            startup_timeout_seconds=max(1, int(runtime.get("startup_timeout_seconds", 120))),
            strict_version=bool(runtime.get("strict_version", True)),
            supported_versions=tuple(str(item) for item in runtime.get("supported_versions", ["0.15"])),
            normal_extra_args=tuple(str(item) for item in normal.get("extra_args", [])),
            allow_simulation_only=bool(embedded.get("allow_simulation_only", False)),
            embedded_extra_args=tuple(str(item) for item in embedded.get("extra_args", [])),
            snapshot_enabled=bool(snapshot.get("enabled", False)),
            gpu_snapshot_enabled=bool(snapshot.get("gpu_enabled", False)),
            resident_models=tuple(str(item) for item in snapshot.get("resident_models", [])),
            json_events=bool(diagnostics.get("json_events", True)),
            event_log=event_log,
        )
