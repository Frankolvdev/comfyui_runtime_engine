from __future__ import annotations

import signal
import subprocess

from .adapters import select_adapter
from .config import RuntimeConfig
from .embedded import EmbeddedBootstrap
from .environment import EnvironmentManager
from .errors import RuntimeEngineError
from .events import EventSink
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
        try:
            import torch
            torch_version = torch.__version__
            cuda_compiled = bool(getattr(torch.version, "cuda", None))
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            torch_version = None
            cuda_compiled = False
            cuda_available = False
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
            "torch_version": torch_version,
            "torch_cuda_compiled": cuda_compiled,
            "torch_cuda_available": cuda_available,
        }
        self.events.emit("doctor.completed", **report)
        return report

    def simulate(self) -> None:
        EmbeddedSimulation(self.inspection, self.events).run()

    def probe_embedded(self) -> dict[str, object]:
        return self._embedded_bootstrap().probe()

    def probe_server(self, hold_seconds: float = 1.0) -> dict[str, object]:
        return self._embedded_bootstrap().server_probe(hold_seconds=hold_seconds)

    def environment_audit(self) -> dict[str, object]:
        return EnvironmentManager(self.config.comfyui_path, self.events).audit()

    def environment_verify(self) -> dict[str, object]:
        return EnvironmentManager(self.config.comfyui_path, self.events).verify()

    def environment_prepare(self) -> dict[str, object]:
        return EnvironmentManager(self.config.comfyui_path, self.events).ensure_workspace()

    def environment_sync(self, *, dry_run: bool = False, strict: bool = False) -> dict[str, object]:
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
