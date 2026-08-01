from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    config_path: Path
    mode: str
    comfyui_path: Path
    python_executable: str
    host: str
    port: int
    startup_timeout_seconds: float
    strict_version: bool
    supported_versions: tuple[str, ...]
    normal_extra_args: tuple[str, ...]
    embedded_extra_args: tuple[str, ...]
    allow_simulation_only: bool
    shutdown_grace_seconds: float
    ensure_workspace: bool
    snapshot_enabled: bool
    gpu_snapshot_enabled: bool
    resident_models: tuple[str, ...]
    model_roots: tuple[Path, ...]
    residency_strict: bool
    execution_reserve_gb: float
    json_events: bool
    event_log: Path | None

    @classmethod
    def from_toml(cls, path: str | Path) -> "RuntimeConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        with config_path.open("rb") as file:
            data = tomllib.load(file)

        runtime = data.get("runtime", {})
        normal = data.get("normal", {})
        embedded = data.get("embedded", {})
        snapshot = data.get("snapshot", {})
        residency = data.get("residency", {})
        diagnostics = data.get("diagnostics", {})

        comfyui_path = Path(str(runtime.get("comfyui_path", ""))).expanduser()
        if not comfyui_path.is_absolute():
            comfyui_path = (config_path.parent / comfyui_path).resolve()
        else:
            comfyui_path = comfyui_path.resolve()

        configured_roots = residency.get("model_roots", [])
        model_roots: list[Path] = []
        for raw_root in configured_roots:
            root = Path(str(raw_root)).expanduser()
            if not root.is_absolute():
                root = (comfyui_path / root).resolve()
            else:
                root = root.resolve()
            model_roots.append(root)

        event_log_value = diagnostics.get("event_log")
        event_log = None
        if event_log_value:
            event_log = Path(str(event_log_value)).expanduser()
            if not event_log.is_absolute():
                event_log = (config_path.parent / event_log).resolve()

        config = cls(
            config_path=config_path,
            mode=str(runtime.get("mode", "normal")),
            comfyui_path=comfyui_path,
            python_executable=str(runtime.get("python_executable", sys.executable)),
            host=str(runtime.get("host", "127.0.0.1")),
            port=int(runtime.get("port", 8188)),
            startup_timeout_seconds=float(runtime.get("startup_timeout_seconds", 120)),
            strict_version=bool(runtime.get("strict_version", True)),
            supported_versions=tuple(
                str(item) for item in runtime.get("supported_versions", ["0.15"])
            ),
            normal_extra_args=tuple(str(item) for item in normal.get("extra_args", [])),
            embedded_extra_args=tuple(
                str(item) for item in embedded.get("extra_args", [])
            ),
            allow_simulation_only=bool(
                embedded.get("allow_simulation_only", True)
            ),
            shutdown_grace_seconds=float(
                embedded.get("shutdown_grace_seconds", 2.0)
            ),
            ensure_workspace=bool(embedded.get("ensure_workspace", True)),
            snapshot_enabled=bool(snapshot.get("enabled", False)),
            gpu_snapshot_enabled=bool(snapshot.get("gpu_enabled", False)),
            resident_models=tuple(
                str(item) for item in snapshot.get("resident_models", [])
            ),
            model_roots=tuple(model_roots),
            residency_strict=bool(residency.get("strict", True)),
            execution_reserve_gb=float(residency.get("execution_reserve_gb", 8.0)),
            json_events=bool(diagnostics.get("json_events", True)),
            event_log=event_log,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in {"normal", "embedded", "embedded-simulation"}:
            raise ConfigurationError(f"Unsupported runtime mode: {self.mode}")
        if not self.comfyui_path.is_dir():
            raise ConfigurationError(
                f"ComfyUI directory does not exist: {self.comfyui_path}"
            )
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("Port must be between 1 and 65535")
        if self.startup_timeout_seconds <= 0:
            raise ConfigurationError("startup_timeout_seconds must be positive")
        if self.shutdown_grace_seconds < 0:
            raise ConfigurationError("shutdown_grace_seconds cannot be negative")
        if self.execution_reserve_gb < 0:
            raise ConfigurationError("residency.execution_reserve_gb cannot be negative")
        if self.gpu_snapshot_enabled and not self.snapshot_enabled:
            raise ConfigurationError(
                "snapshot.gpu_enabled requires snapshot.enabled=true"
            )
