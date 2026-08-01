from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from ..events import EventSink


class IsolatedGPUProbe:
    """Run CUDA diagnostics in a disposable Python subprocess.

    This guarantees that the parent process can later import ComfyUI before
    importing torch, which is required by ComfyUI's startup contract.
    """

    def __init__(self, events: EventSink) -> None:
        self.events = events

    def run(self, timeout_seconds: float = 60.0) -> dict[str, Any]:
        helper = Path(__file__).with_name("_probe_worker.py")
        command = [sys.executable, str(helper)]
        self.events.emit(
            "gpu.isolated_probe.started",
            command=command,
            parent_pid=os.getpid(),
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        try:
            report = json.loads(stdout)
        except json.JSONDecodeError:
            report = {
                "success": False,
                "error": "GPU probe subprocess did not return valid JSON.",
                "stdout": stdout,
                "stderr": stderr,
            }

        report["isolated_process"] = True
        report["parent_pid"] = os.getpid()
        report["return_code"] = completed.returncode
        if stderr:
            report["stderr"] = stderr
        report["success"] = bool(report.get("success")) and completed.returncode == 0
        self.events.emit("gpu.isolated_probe.completed", **report)
        return report
