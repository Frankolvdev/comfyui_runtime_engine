from pathlib import Path

from comfyui_runtime_engine.environment import EnvironmentManager
from comfyui_runtime_engine.events import EventSink


def test_environment_repair_dry_run_is_non_destructive(tmp_path: Path) -> None:
    manager = EnvironmentManager(tmp_path, EventSink(tmp_path / "events.jsonl"))
    report = manager.repair(dry_run=True)
    assert report["success"] is True
    assert len(report["commands"]) == 2
    assert "opencv-contrib-python==4.11.0.86" in report["commands"][1]
    assert any(str(item).startswith("diffusers>=") for item in report["commands"][1])
