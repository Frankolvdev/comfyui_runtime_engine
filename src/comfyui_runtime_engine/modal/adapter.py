from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from .contracts import AtomicJsonFile, ModalHealthReport, ModalRuntimePaths


class ModalSnapshotAdapter:
    """Provider-facing adapter used by generated Modal applications.

    The adapter deliberately does not import the ``modal`` package. The backend
    owns Modal decorators and existing job/cancellation behavior; this class
    only supplies lifecycle methods for those decorators.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        host: str = "127.0.0.1",
        port: int = 8000,
        startup_timeout_seconds: float = 900.0,
        ready_path: str | Path = "/tmp/comfy-runtime-modal-ready.json",
        restore_path: str | Path = "/tmp/comfy-runtime-modal-restore.json",
        log_path: str | Path = "/tmp/comfy-runtime-modal.log",
        python_executable: str | None = None,
    ) -> None:
        self.paths = ModalRuntimePaths.from_values(
            config_path=config_path,
            ready_path=ready_path,
            restore_path=restore_path,
            log_path=log_path,
        )
        self.host = host
        self.port = int(port)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.python_executable = python_executable or sys.executable
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def prepare_snapshot(self) -> dict[str, Any]:
        """Start the persistent runtime and wait for snapshot readiness.

        Intended to be called from ``@modal.enter(snap=True)``.
        """
        self.paths.ready_path.unlink(missing_ok=True)
        self.paths.restore_path.unlink(missing_ok=True)
        self.paths.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.paths.log_path.open("a", encoding="utf-8")

        command = [
            self.python_executable,
            "-m",
            "comfyui_runtime_engine",
            "--config",
            str(self.paths.config_path),
            "modal",
            "serve",
            "--ready-path",
            str(self.paths.ready_path),
        ]
        env = os.environ.copy()
        env["COMFY_RUNTIME_MODAL"] = "1"
        env["COMFY_RUNTIME_MODAL_HOST"] = self.host
        env["COMFY_RUNTIME_MODAL_PORT"] = str(self.port)

        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

        state = AtomicJsonFile(self.paths.ready_path).wait(
            timeout_seconds=self.startup_timeout_seconds,
            predicate=lambda item: bool(item.get("snapshot_ready")),
        )
        self._assert_process_alive()
        self._wait_http_ready(self.startup_timeout_seconds)

        report = self.health_check(phase="before_snapshot")
        if not report["success"]:
            raise RuntimeError(
                "ComfyUI runtime did not reach snapshot-ready state: "
                + json.dumps(report, ensure_ascii=False)
            )
        return report

    def after_restore(self) -> dict[str, Any]:
        """Validate child process, HTTP service and resident manifest.

        Intended to be called from ``@modal.enter(snap=False)``. This does not
        reload model weights. It fails closed if the restored runtime is not
        healthy, preserving the existing backend pipeline from accepting work
        against a partial restore.
        """
        self._assert_process_alive()
        self._wait_http_ready(self.startup_timeout_seconds)
        report = self.health_check(phase="after_restore")
        AtomicJsonFile(self.paths.restore_path).write(report)
        if not report["success"]:
            raise RuntimeError(
                "Restored ComfyUI runtime failed validation: "
                + json.dumps(report, ensure_ascii=False)
            )
        return report

    def health_check(self, *, phase: str = "runtime") -> dict[str, Any]:
        state = AtomicJsonFile(self.paths.ready_path).read()
        expected = int(state.get("expected_resident_count", 0))
        actual = int(state.get("resident_count", 0))
        server_ready = self._http_ready()
        snapshot_ready = bool(state.get("snapshot_ready"))
        process_alive = self.process is not None and self.process.poll() is None

        success = bool(
            process_alive
            and server_ready
            and snapshot_ready
            and expected > 0
            and actual == expected
        )
        report = ModalHealthReport(
            success=success,
            phase=phase,
            pid=self.process.pid if self.process is not None else state.get("pid"),
            server_ready=server_ready,
            snapshot_ready=snapshot_ready,
            resident_count=actual,
            expected_resident_count=expected,
            details={
                "process_alive": process_alive,
                "ready_path": str(self.paths.ready_path),
                "restore_path": str(self.paths.restore_path),
                "log_path": str(self.paths.log_path),
                "resident_identities": state.get("resident_identities", []),
                "vram": state.get("vram", {}),
                "workflow": state.get("workflow"),
                "prompt_id": state.get("prompt_id"),
            },
        )
        return report.to_dict()

    def terminate(self, *, timeout_seconds: float = 20.0) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _assert_process_alive(self) -> None:
        if self.process is None:
            raise RuntimeError("Modal runtime process has not been started.")
        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Modal runtime process exited with code {return_code}. "
                f"See {self.paths.log_path}."
            )

    def _wait_http_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self._assert_process_alive()
            if self._http_ready():
                return
            time.sleep(0.25)
        raise TimeoutError(
            f"ComfyUI HTTP server did not become ready on "
            f"{self.host}:{self.port}."
        )

    def _http_ready(self) -> bool:
        urls = (
            f"http://{self.host}:{self.port}/system_stats",
            f"http://{self.host}:{self.port}/",
        )
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if 200 <= int(response.status) < 500:
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False
