from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .errors import ConfigurationError
from .warmup import WarmupInput


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
    warmup_workflow: Path | None
    warmup_timeout_seconds: float
    warmup_inputs: tuple[WarmupInput, ...]
    sam3_source_model: Path | None
    sam3_expected_model: Path
    sam3_link_mode: str
    sam3_replace_existing: bool
    snapshot_provider: str
    snapshot_state_path: Path
    snapshot_audit_path: Path
    json_events: bool
    event_log: Path | None

    @classmethod
    def from_toml(cls, path: str | Path) -> "RuntimeConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        with config_path.open("rb") as file:
            raw = tomllib.load(file)

        runtime = raw.get("runtime", {})
        normal = raw.get("normal", {})
        embedded = raw.get("embedded", {})
        snapshot = raw.get("snapshot", {})
        residency = raw.get("residency", {})
        diagnostics = raw.get("diagnostics", {})
        snapshot_lifecycle = raw.get("snapshot_lifecycle", {})

        comfyui_path = Path(str(runtime["comfyui_path"])).expanduser()
        comfyui_path = (
            (config_path.parent / comfyui_path).resolve()
            if not comfyui_path.is_absolute()
            else comfyui_path.resolve()
        )

        model_roots: list[Path] = []
        for value in residency.get("model_roots", []):
            root = Path(str(value)).expanduser()
            root = (
                (comfyui_path / root).resolve()
                if not root.is_absolute()
                else root.resolve()
            )
            model_roots.append(root)

        workflow = None
        if residency.get("warmup_workflow"):
            workflow = Path(str(residency["warmup_workflow"])).expanduser()
            workflow = (
                (config_path.parent / workflow).resolve()
                if not workflow.is_absolute()
                else workflow.resolve()
            )

        warmup_inputs: list[WarmupInput] = []
        for item in residency.get("warmup_inputs", []):
            if not isinstance(item, dict):
                raise ConfigurationError(
                    "Each residency.warmup_inputs item must be an inline TOML object"
                )
            source = Path(str(item.get("source", ""))).expanduser()
            source = (
                (config_path.parent / source).resolve()
                if not source.is_absolute()
                else source.resolve()
            )
            warmup_inputs.append(
                WarmupInput(source=source, target=str(item.get("target", "")).strip())
            )

        first_root = model_roots[0] if model_roots else comfyui_path / "models"
        source_value = residency.get("sam3_source_model")
        if source_value:
            sam3_source = Path(str(source_value)).expanduser()
            sam3_source = (
                (config_path.parent / sam3_source).resolve()
                if not sam3_source.is_absolute()
                else sam3_source.resolve()
            )
        else:
            sam3_source = (first_root / "sam3" / "sam3.pt").resolve()

        expected_value = residency.get("sam3_expected_model")
        if expected_value:
            sam3_expected = Path(str(expected_value)).expanduser()
            sam3_expected = (
                (comfyui_path / sam3_expected).absolute()
                if not sam3_expected.is_absolute()
                else sam3_expected.absolute()
            )
        else:
            sam3_expected = (
                comfyui_path / "models" / "sam3" / "sam3.pt"
            ).absolute()

        event_log = Path(
            str(diagnostics.get("event_log", ".runtime/events.jsonl"))
        ).expanduser()
        if not event_log.is_absolute():
            event_log = (config_path.parent / event_log).resolve()

        state_path = Path(
            str(snapshot_lifecycle.get("state_path", ".runtime/snapshot-state.json"))
        ).expanduser()
        if not state_path.is_absolute():
            state_path = (config_path.parent / state_path).resolve()

        audit_path = Path(
            str(snapshot_lifecycle.get("audit_path", ".runtime/snapshot-audit.json"))
        ).expanduser()
        if not audit_path.is_absolute():
            audit_path = (config_path.parent / audit_path).resolve()

        config = cls(
            config_path=config_path,
            mode=str(runtime.get("mode", "embedded-simulation")),
            comfyui_path=comfyui_path,
            python_executable=str(runtime.get("python_executable", sys.executable)),
            host=str(runtime.get("host", "127.0.0.1")),
            port=int(runtime.get("port", 8188)),
            startup_timeout_seconds=float(runtime.get("startup_timeout_seconds", 120)),
            strict_version=bool(runtime.get("strict_version", True)),
            supported_versions=tuple(
                map(str, runtime.get("supported_versions", ["0.15"]))
            ),
            normal_extra_args=tuple(map(str, normal.get("extra_args", []))),
            embedded_extra_args=tuple(map(str, embedded.get("extra_args", []))),
            allow_simulation_only=bool(
                embedded.get("allow_simulation_only", False)
            ),
            shutdown_grace_seconds=float(
                embedded.get("shutdown_grace_seconds", 2)
            ),
            ensure_workspace=bool(embedded.get("ensure_workspace", True)),
            snapshot_enabled=bool(snapshot.get("enabled", False)),
            gpu_snapshot_enabled=bool(snapshot.get("gpu_enabled", False)),
            resident_models=tuple(map(str, snapshot.get("resident_models", []))),
            model_roots=tuple(model_roots),
            residency_strict=bool(residency.get("strict", True)),
            execution_reserve_gb=float(
                residency.get("execution_reserve_gb", 8)
            ),
            warmup_workflow=workflow,
            warmup_timeout_seconds=float(
                residency.get("warmup_timeout_seconds", 600)
            ),
            warmup_inputs=tuple(warmup_inputs),
            sam3_source_model=sam3_source,
            sam3_expected_model=sam3_expected,
            sam3_link_mode=str(residency.get("sam3_link_mode", "auto")),
            sam3_replace_existing=bool(
                residency.get("sam3_replace_existing", False)
            ),
            snapshot_provider=str(snapshot_lifecycle.get("provider", "modal")),
            snapshot_state_path=state_path,
            snapshot_audit_path=audit_path,
            json_events=bool(diagnostics.get("json_events", True)),
            event_log=event_log,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.comfyui_path.is_dir():
            raise ConfigurationError(
                f"ComfyUI directory does not exist: {self.comfyui_path}"
            )
        if self.sam3_link_mode not in {"auto", "symlink", "hardlink", "copy"}:
            raise ConfigurationError(
                "residency.sam3_link_mode must be auto, symlink, hardlink or copy"
            )
        if self.snapshot_provider not in {"modal", "local-simulation"}:
            raise ConfigurationError(
                "snapshot_lifecycle.provider must be modal or local-simulation"
            )
        for item in self.warmup_inputs:
            item.validate()
