from __future__ import annotations

import signal
import subprocess
import sys
import time

from .adapters import select_adapter
from .config import RuntimeConfig
from .embedded import EmbeddedBootstrap, EmbeddedWarmupRunner, local_probe_environment
from .environment import EnvironmentManager
from .errors import RuntimeEngineError
from .events import EventSink
from .gpu import CUDAEnvironmentSetup, GPUProbe, IsolatedGPUProbe
from .lifecycle import ResourceMonitor
from .residency import ResidencyManager
from .simulation import EmbeddedSimulation
from .warmup import WarmupWorkflow


class RuntimeEngine:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.events = EventSink(config.event_log if config.json_events else None)
        self.adapter, self.inspection = select_adapter(
            config.comfyui_path, config.supported_versions, config.strict_version
        )

    def doctor(self) -> dict[str, object]:
        gpu = IsolatedGPUProbe(self.events).run()
        report = {
            "root": str(self.inspection.root),
            "version": self.inspection.version,
            "adapter": self.adapter.family,
            "structurally_valid": self.inspection.structurally_valid,
            "mode": self.config.mode,
            "snapshot_enabled": self.config.snapshot_enabled,
            "gpu_snapshot_enabled": self.config.gpu_snapshot_enabled,
            "resident_models": list(self.config.resident_models),
            "model_roots": [str(p) for p in self._effective_model_roots()],
            "warmup_workflow": (
                str(self.config.warmup_workflow)
                if self.config.warmup_workflow else None
            ),
            "gpu": gpu,
            "torch_imported_in_parent": False,
        }
        self.events.emit("doctor.completed", **report)
        return report

    def residency_scan(self): return self._residency_manager().scan()
    def residency_plan(self): return self._residency_manager().plan()
    def residency_verify(self): return self._residency_manager().verify()

    def warmup_verify(self) -> dict[str, object]:
        if self.config.warmup_workflow is None:
            raise RuntimeEngineError("residency.warmup_workflow is not configured")
        workflow = WarmupWorkflow.load(self.config.warmup_workflow)
        residency = self.residency_verify()
        report = {
            "workflow": str(workflow.path),
            "node_count": workflow.node_count,
            "residency": residency,
            "success": bool(residency["success"]),
        }
        self.events.emit("warmup.verify.completed", **report)
        return report

    def warmup_probe(self, hold_seconds: float = 5.0) -> dict[str, object]:
        gpu = self.gpu_probe(isolated=True)
        if not gpu.get("success"):
            raise RuntimeEngineError("GPU verification failed before warmup")
        self.warmup_verify()
        gpu_args = tuple(a for a in self.config.embedded_extra_args if a != "--cpu")
        bootstrap = self._embedded_bootstrap(extra_args=gpu_args)
        workflow = WarmupWorkflow.load(self.config.warmup_workflow)
        report = EmbeddedWarmupRunner(bootstrap).run(
            workflow,
            timeout_seconds=self.config.warmup_timeout_seconds,
            hold_seconds=hold_seconds,
        )
        report["gpu_probe"] = gpu
        report["resident_plan"] = self.residency_plan()
        return report

    def _residency_manager(self):
        return ResidencyManager(
            comfyui_root=self.config.comfyui_path,
            model_roots=self._effective_model_roots(),
            resident_models=self.config.resident_models,
            events=self.events,
        )

    def _effective_model_roots(self):
        return self.config.model_roots or (self.config.comfyui_path / "models",)

    def simulate(self): EmbeddedSimulation(self.inspection, self.events).run()

    def gpu_probe(self, isolated=True):
        return IsolatedGPUProbe(self.events).run() if isolated else GPUProbe(self.events).run()

    def gpu_setup(self, dry_run=False, channel="cu128"):
        return CUDAEnvironmentSetup(self.events).prepare(dry_run=dry_run, channel=channel)

    def repeated_gpu_server_probe(self, iterations=2, hold_seconds=1.0):
        command = [
            sys.executable, "-m", "comfyui_runtime_engine", "--config",
            str(self.config.config_path), "gpu-server-probe",
            "--hold-seconds", str(hold_seconds),
        ]
        results = []
        for i in range(iterations):
            cp = subprocess.run(command, cwd=self.config.config_path.parent,
                capture_output=True, text=True, check=False,
                timeout=max(180.0, self.config.startup_timeout_seconds + 60))
            results.append({"iteration": i+1, "return_code": cp.returncode,
                "stdout_tail": cp.stdout[-5000:], "stderr_tail": cp.stderr[-5000:],
                "success": cp.returncode == 0})
        return {"iterations": iterations, "python_executable": sys.executable,
            "command": command, "results": results,
            "success": all(x["success"] for x in results)}

    def probe_gpu_server(self, hold_seconds=2.0):
        gpu = self.gpu_probe(True)
        if not gpu.get("success"): raise RuntimeEngineError("GPU probe failed")
        args = tuple(a for a in self.config.embedded_extra_args if a != "--cpu")
        report = self._embedded_bootstrap(extra_args=args).server_probe(
            hold_seconds=hold_seconds)
        report["gpu_probe"] = gpu
        report["cpu_flag_removed"] = "--cpu" not in args
        return report

    def environment_audit(self):
        return EnvironmentManager(self.config.comfyui_path, self.events).audit()
    def environment_verify(self):
        return EnvironmentManager(self.config.comfyui_path, self.events).verify()
    def environment_prepare(self):
        return EnvironmentManager(self.config.comfyui_path, self.events).ensure_workspace()
    def environment_repair(self, dry_run=False):
        return EnvironmentManager(self.config.comfyui_path, self.events).repair(dry_run=dry_run)
    def environment_sync(self, dry_run=False, strict=False):
        return EnvironmentManager(self.config.comfyui_path, self.events).sync(
            dry_run=dry_run, strict=strict).to_dict()

    def _embedded_bootstrap(self, extra_args=None):
        manager = EnvironmentManager(self.config.comfyui_path, self.events)
        if self.config.ensure_workspace: manager.ensure_workspace()
        return EmbeddedBootstrap(
            self.inspection, self.events, host=self.config.host, port=self.config.port,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            extra_args=self.config.embedded_extra_args if extra_args is None else extra_args,
            shutdown_grace_seconds=self.config.shutdown_grace_seconds)

    def start(self, mode=None):
        selected = (mode or self.config.mode).lower()
        if selected == "embedded-simulation": self.simulate(); return 0
        if selected == "embedded": return self._embedded_bootstrap().serve()
        if selected != "normal": raise RuntimeEngineError(f"Unsupported mode: {selected}")
        command = self.adapter.normal_command(
            root=self.config.comfyui_path,
            python_executable=self.config.python_executable,
            host=self.config.host, port=self.config.port,
            extra_args=self.config.normal_extra_args)
        return subprocess.call(command, cwd=self.config.comfyui_path)
