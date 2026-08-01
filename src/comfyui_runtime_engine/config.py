from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys, tomllib
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
    supported_versions: tuple[str,...]
    normal_extra_args: tuple[str,...]
    embedded_extra_args: tuple[str,...]
    allow_simulation_only: bool
    shutdown_grace_seconds: float
    ensure_workspace: bool
    snapshot_enabled: bool
    gpu_snapshot_enabled: bool
    resident_models: tuple[str,...]
    model_roots: tuple[Path,...]
    residency_strict: bool
    execution_reserve_gb: float
    warmup_workflow: Path|None
    warmup_timeout_seconds: float
    json_events: bool
    event_log: Path|None

    @classmethod
    def from_toml(cls,path):
        cp=Path(path).expanduser().resolve()
        with cp.open("rb") as f: raw=tomllib.load(f)
        r=raw.get("runtime",{}); n=raw.get("normal",{}); e=raw.get("embedded",{})
        s=raw.get("snapshot",{}); z=raw.get("residency",{}); d=raw.get("diagnostics",{})
        root=Path(str(r["comfyui_path"])).expanduser()
        root=(cp.parent/root).resolve() if not root.is_absolute() else root.resolve()
        roots=[]
        for value in z.get("model_roots",[]):
            p=Path(str(value)).expanduser()
            roots.append((root/p).resolve() if not p.is_absolute() else p.resolve())
        wf=None
        if z.get("warmup_workflow"):
            wf=Path(str(z["warmup_workflow"])).expanduser()
            wf=(cp.parent/wf).resolve() if not wf.is_absolute() else wf.resolve()
        log=Path(str(d.get("event_log",".runtime/events.jsonl"))).expanduser()
        if not log.is_absolute(): log=(cp.parent/log).resolve()
        return cls(cp,str(r.get("mode","embedded-simulation")),root,
            str(r.get("python_executable",sys.executable)),str(r.get("host","127.0.0.1")),
            int(r.get("port",8188)),float(r.get("startup_timeout_seconds",120)),
            bool(r.get("strict_version",True)),tuple(map(str,r.get("supported_versions",["0.15"]))),
            tuple(map(str,n.get("extra_args",[]))),tuple(map(str,e.get("extra_args",[]))),
            bool(e.get("allow_simulation_only",False)),float(e.get("shutdown_grace_seconds",2)),
            bool(e.get("ensure_workspace",True)),bool(s.get("enabled",False)),
            bool(s.get("gpu_enabled",False)),tuple(map(str,s.get("resident_models",[]))),
            tuple(roots),bool(z.get("strict",True)),float(z.get("execution_reserve_gb",8)),
            wf,float(z.get("warmup_timeout_seconds",600)),
            bool(d.get("json_events",True)),log)
