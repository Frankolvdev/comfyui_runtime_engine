from __future__ import annotations

import signal
import subprocess
import sys
import time

from .adapters import select_adapter
from .config import RuntimeConfig
from .embedded import EmbeddedBootstrap, local_probe_environment
from .environment import EnvironmentManager
from .errors import RuntimeEngineError
from .events import EventSink
from .gpu import CUDAEnvironmentSetup, GPUProbe, IsolatedGPUProbe
from .lifecycle import ResourceMonitor
from .residency import ResidencyManager
from .simulation import EmbeddedSimulation


class RuntimeEngine:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.events = EventSink(config.event_log if config.json_events else None)
        self.adapter, self.inspection = select_adapter(
            config.comfyui_path,
            config.supported_versions,
            config.strict_version,
        )

    def doctor(self) -> dict[str, object]:
        gpu = IsolatedGPUProbe(self.events).run()
        report: dict[str, object] = {
            "root": str(self.inspection.root),
            "version": self.inspection.version,
            "adapter": self.adapter.family,
            "structurally_valid": self.inspection.structurally_valid,
            "mode": self.config.mode,
            "embedded_entrypoint": "main.start_comfyui",
            "snapshot_enabled": self.config.snapshot_enabled,
            "gpu_snapshot_enabled": self.config.gpu_snapshot_enabled,
            "resident_models": list(self.config.resident_models),
            "model_roots": [str(path) for path in self._effective_model_roots()],
            "gpu": gpu,
            "torch_imported_in_parent": False,
        }
        self.events.emit("doctor.completed", **report)
        return report

    def residency_scan(self) -> dict[str, object]:
        return self._residency_manager().scan()

    def residency_plan(self) -> dict[str, object]:
        return self._residency_manager().plan()

    def residency_verify(self) -> dict[str, object]:
        return self._residency_manager().verify()

    def _residency_manager(self) -> ResidencyManager:
        return ResidencyManager(
            comfyui_root=self.config.comfyui_path,
            model_roots=self._effective_model_roots(),
            resident_models=self.config.resident_models,
            events=self.events,
        )

    def _effective_model_roots(self) -> tuple:
        return self.config.model_roots or (self.config.comfyui_path / "models",)

    def simulate(self) -> None:
        EmbeddedSimulation(self.inspection, self.events).run()

    def probe_embedded(self) -> dict[str, object]:
        before = ResourceMonitor.snapshot()
        with local_probe_environment(True):
            report = self._embedded_bootstrap().probe()
        time.sleep(self.config.shutdown_grace_seconds)
        ResourceMonitor.cleanup_gc()
        after = ResourceMonitor.snapshot()
        report["resources"] = ResourceMonitor.compare(before, after)
        return report

    def probe_server(self, hold_seconds: float = 1.0) -> dict[str, object]:
        before = ResourceMonitor.snapshot()
        with local_probe_environment(True):
            report = self._embedded_bootstrap().server_probe(
                hold_seconds=hold_seconds
            )
        time.sleep(self.config.shutdown_grace_seconds)
        ResourceMonitor.cleanup_gc()
        after = ResourceMonitor.snapshot()
        report["resources"] = ResourceMonitor.compare(before, after)
        return report

    def probe_gpu_server(self, hold_seconds: float = 2.0) -> dict[str, object]:
        gpu_report = self.gpu_probe(isolated=True)
        if not bool(gpu_report.get("success")):
            raise RuntimeEngineError(
                "GPU probe failed. Run 'gpu setup' and then 'gpu-probe'."
            )
        manager = EnvironmentManager(self.config.comfyui_path, self.events)
        if self.config.ensure_workspace:
            manager.ensure_workspace()

        gpu_args = tuple(arg for arg in self.config.embedded_extra_args if arg != "--cpu")
        bootstrap = EmbeddedBootstrap(
            self.inspection,
            self.events,
            host=self.config.host,
            port=self.config.port,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            extra_args=gpu_args,
            shutdown_grace_seconds=self.config.shutdown_grace_seconds,
        )

        before = ResourceMonitor.snapshot()
        with local_probe_environment(True):
            report = bootstrap.server_probe(hold_seconds=hold_seconds)
        time.sleep(self.config.shutdown_grace_seconds)
        ResourceMonitor.cleanup_gc()
        after = ResourceMonitor.snapshot()

        report["gpu_probe"] = gpu_report
        report["cpu_flag_removed"] = "--cpu" not in gpu_args
        report["torch_imported_in_parent_before_comfyui"] = False
        report["resources"] = ResourceMonitor.compare(before, after)
        return report

    def repeated_gpu_server_probe(
        self,
        *,
        iterations: int = 2,
        hold_seconds: float = 1.0,
    ) -> dict[str, object]:
        if iterations < 1 or iterations > 10:
            raise RuntimeEngineError("iterations must be between 1 and 10")

        command = [
            sys.executable,
            "-m",
            "comfyui_runtime_engine",
            "--config",
            str(self.config.config_path),
            "gpu-server-probe",
            "--hold-seconds",
            str(hold_seconds),
        ]
        results: list[dict[str, object]] = []
        for index in range(iterations):
            completed = subprocess.run(
                command,
                cwd=self.config.config_path.parent,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(180.0, self.config.startup_timeout_seconds + 60.0),
            )
            results.append(
                {
                    "iteration": index + 1,
                    "return_code": completed.returncode,
                    "stdout_tail": completed.stdout[-5000:],
                    "stderr_tail": completed.stderr[-5000:],
                    "success": completed.returncode == 0,
                }
            )
        report = {
            "iterations": iterations,
            "python_executable": sys.executable,
            "command": command,
            "results": results,
            "success": all(bool(item["success"]) for item in results),
        }
        self.events.emit("gpu.repeated_server_probe.completed", **report)
        return report

    def gpu_probe(self, *, isolated: bool = True) -> dict[str, object]:
        if isolated:
            return IsolatedGPUProbe(self.events).run()
        return GPUProbe(self.events).run()

    def gpu_setup(self, *, dry_run: bool = False, channel: str = "cu128") -> dict[str, object]:
        return CUDAEnvironmentSetup(self.events).prepare(
            dry_run=dry_run,
            channel=channel,
        )

    def environment_audit(self) -> dict[str, object]:
        return EnvironmentManager(self.config.comfyui_path, self.events).audit()

    def environment_verify(self) -> dict[str, object]:
        return EnvironmentManager(self.config.comfyui_path, self.events).verify()

    def environment_prepare(self) -> dict[str, object]:
        return EnvironmentManager(self.config.comfyui_path, self.events).ensure_workspace()

    def environment_repair(self, *, dry_run: bool = False) -> dict[str, object]:
        return EnvironmentManager(self.config.comfyui_path, self.events).repair(
            dry_run=dry_run
        )

    def environment_sync(
        self,
        *,
        dry_run: bool = False,
        strict: bool = False,
    ) -> dict[str, object]:
        return EnvironmentManager(self.config.comfyui_path, self.events).sync(
            dry_run=dry_run,
            strict=strict,
        ).to_dict()

    def start(self, mode: str | None = None) -> int:
        selected_mode = (mode or self.config.mode).lower()
        if selected_mode == "embedded-simulation":
            self.simulate()
            return 0
        if selected_mode == "embedded":
            if self.config.allow_simulation_only:
                raise RuntimeEngineError(
                    "Embedded mode is disabled by embedded.allow_simulation_only=true"
                )
            return self._embedded_bootstrap().serve()
        if selected_mode != "normal":
            raise RuntimeEngineError(f"Unsupported start mode: {selected_mode}")
        return self._start_normal()

    def _embedded_bootstrap(self) -> EmbeddedBootstrap:
        manager = EnvironmentManager(self.config.comfyui_path, self.events)
        if self.config.ensure_workspace:
            manager.ensure_workspace()
        return EmbeddedBootstrap(
            self.inspection,
            self.events,
            host=self.config.host,
            port=self.config.port,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            extra_args=self.config.embedded_extra_args,
            shutdown_grace_seconds=self.config.shutdown_grace_seconds,
        )

    def _start_normal(self) -> int:
        command = self.adapter.normal_command(
            root=self.config.comfyui_path,
            python_executable=self.config.python_executable,
            host=self.config.host,
            port=self.config.port,
            extra_args=self.config.normal_extra_args,
        )
        self.events.emit("normal.starting", command=command, cwd=str(self.config.comfyui_path))
        process = subprocess.Popen(command, cwd=self.config.comfyui_path)

        def terminate(_signum: int, _frame: object) -> None:
            if process.poll() is None:
                self.events.emit("normal.terminating", pid=process.pid)
                process.terminate()

        previous_sigint = signal.signal(signal.SIGINT, terminate)
        previous_sigterm = signal.signal(signal.SIGTERM, terminate)
        try:
            return_code = process.wait()
            self.events.emit("normal.exited", pid=process.pid, return_code=return_code)
            return return_code
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
