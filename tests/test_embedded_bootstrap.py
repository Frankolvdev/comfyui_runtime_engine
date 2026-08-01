import os
from pathlib import Path

from comfyui_runtime_engine.adapters import select_adapter
from comfyui_runtime_engine.embedded import EmbeddedBootstrap
from comfyui_runtime_engine.events import EventSink


def test_real_embedded_probe_stays_in_engine_process(tmp_path: Path) -> None:
    root = Path(__file__).parent / "fixtures" / "fake_comfyui"
    _, inspection = select_adapter(root, ("0.15",), True)
    bootstrap = EmbeddedBootstrap(
        inspection,
        EventSink(tmp_path / "events.jsonl"),
        host="127.0.0.1",
        port=8188,
    )
    report = bootstrap.probe()
    assert report["same_process"] is True
    assert report["pid"] == os.getpid()
    assert report["prompt_server"] == "FakePromptServer"
    assert report["prompt_queue"] == "FakeQueue"
    assert report["server_bound"] is False
