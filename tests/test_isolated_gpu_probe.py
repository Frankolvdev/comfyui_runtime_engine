from __future__ import annotations

import json
from pathlib import Path
import subprocess

from comfyui_runtime_engine.events import EventSink
from comfyui_runtime_engine.gpu.isolated_probe import IsolatedGPUProbe


def test_isolated_probe_uses_subprocess(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "success": True,
        "torch_imported": True,
        "cuda_compiled": True,
        "cuda_available": True,
        "device_count": 1,
        "allocation_test": True,
    }

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = IsolatedGPUProbe(EventSink(tmp_path / "events.jsonl")).run()
    assert report["success"] is True
    assert report["isolated_process"] is True
