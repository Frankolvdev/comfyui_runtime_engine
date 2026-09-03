from __future__ import annotations

from pathlib import Path

import pytest

from comfyui_runtime_engine.modal import AtomicJsonFile, ModalSnapshotAdapter


class ExitedProcess:
    pid = 321

    def poll(self):
        return 7


class AliveProcess:
    pid = 654

    def poll(self):
        return None


def test_wait_snapshot_ready_fails_immediately_with_log_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    log_path.write_text("engine bootstrap failed: sentinel-error\n", encoding="utf-8")
    adapter = ModalSnapshotAdapter(
        config_path=tmp_path / "runtime-engine.toml",
        ready_path=tmp_path / "ready.json",
        log_path=log_path,
        startup_timeout_seconds=30,
    )
    adapter.process = ExitedProcess()

    with pytest.raises(RuntimeError) as exc_info:
        adapter._wait_snapshot_ready(poll_seconds=0.001)

    message = str(exc_info.value)
    assert "exited with code 7" in message
    assert "sentinel-error" in message


def test_wait_snapshot_ready_relays_log_and_returns_state(
    tmp_path: Path,
    capsys,
) -> None:
    ready_path = tmp_path / "ready.json"
    log_path = tmp_path / "runtime.log"
    log_path.write_text("loading resident model\n", encoding="utf-8")
    AtomicJsonFile(ready_path).write(
        {
            "snapshot_ready": True,
            "resident_count": 2,
            "expected_resident_count": 2,
        }
    )
    adapter = ModalSnapshotAdapter(
        config_path=tmp_path / "runtime-engine.toml",
        ready_path=ready_path,
        log_path=log_path,
        startup_timeout_seconds=1,
    )
    adapter.process = AliveProcess()

    state = adapter._wait_snapshot_ready(poll_seconds=0.001)

    assert state["snapshot_ready"] is True
    assert state["resident_count"] == 2
    assert "[comfy-runtime-modal] loading resident model" in capsys.readouterr().out
