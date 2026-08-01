from pathlib import Path

from comfyui_runtime_engine.environment import EnvironmentManager
from comfyui_runtime_engine.events import EventSink


def test_environment_discovers_core_and_custom_node_requirements(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("sqlalchemy\n", encoding="utf-8")
    node = tmp_path / "custom_nodes" / "example-node"
    node.mkdir(parents=True)
    (node / "requirements.txt").write_text("piexif\n", encoding="utf-8")
    manager = EnvironmentManager(tmp_path, EventSink(tmp_path / "events.jsonl"))
    discovered = manager.discover()
    assert discovered == [tmp_path / "requirements.txt", node / "requirements.txt"]
    report = manager.sync(dry_run=True)
    assert report.success is True
    assert len(report.results) == 1
    command = report.results[0].command
    assert command.count("-r") == 2


def test_environment_prepares_writable_workspace(tmp_path: Path) -> None:
    manager = EnvironmentManager(tmp_path, EventSink(tmp_path / "events.jsonl"))
    report = manager.ensure_workspace()
    assert report["writable"] is True
    for name in ("models", "input", "output", "temp", "user"):
        assert (tmp_path / name).is_dir()
