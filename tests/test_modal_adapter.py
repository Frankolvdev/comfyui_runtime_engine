from __future__ import annotations

import json
from pathlib import Path
import subprocess

from comfyui_runtime_engine.modal import (
    AtomicJsonFile,
    ModalSnapshotAdapter,
)


class FakeProcess:
    pid = 123

    def poll(self):
        return None


def test_health_requires_exact_resident_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ready = tmp_path / "ready.json"
    AtomicJsonFile(ready).write(
        {
            "snapshot_ready": True,
            "resident_count": 1,
            "expected_resident_count": 1,
            "resident_identities": [{"identity": "abc"}],
        }
    )
    adapter = ModalSnapshotAdapter(
        config_path=tmp_path / "runtime-engine.toml",
        ready_path=ready,
        restore_path=tmp_path / "restore.json",
        log_path=tmp_path / "runtime.log",
    )
    adapter.process = FakeProcess()
    monkeypatch.setattr(adapter, "_http_ready", lambda: True)

    report = adapter.health_check(phase="after_restore")

    assert report["success"] is True
    assert report["resident_count"] == 1


def test_health_fails_partial_restore(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ready = tmp_path / "ready.json"
    AtomicJsonFile(ready).write(
        {
            "snapshot_ready": True,
            "resident_count": 0,
            "expected_resident_count": 1,
        }
    )
    adapter = ModalSnapshotAdapter(
        config_path=tmp_path / "runtime-engine.toml",
        ready_path=ready,
    )
    adapter.process = FakeProcess()
    monkeypatch.setattr(adapter, "_http_ready", lambda: True)

    assert adapter.health_check()["success"] is False
