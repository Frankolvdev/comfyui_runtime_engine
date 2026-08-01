from pathlib import Path

from comfyui_runtime_engine.environment import EnvironmentManager
from comfyui_runtime_engine.events import EventSink


def test_repair_uses_huggingface_hub_below_one(tmp_path: Path) -> None:
    manager = EnvironmentManager(tmp_path, EventSink(tmp_path / "events.jsonl"))
    report = manager.repair(dry_run=True)
    install = report["commands"][1]
    assert "huggingface_hub>=0.34.0,<1.0" in install
    assert "transformers>=4.57.0,<5" in install


def test_workspace_recreates_temp(tmp_path: Path) -> None:
    manager = EnvironmentManager(tmp_path, EventSink(tmp_path / "events.jsonl"))
    first = manager.ensure_workspace()
    assert first["writable"] is True
    temp = tmp_path / "temp"
    temp.rmdir()
    second = manager.ensure_workspace()
    assert second["paths"]["temp"] is True
    assert second["writable"] is True
