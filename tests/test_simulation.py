from pathlib import Path

from comfyui_runtime_engine.adapters import select_adapter
from comfyui_runtime_engine.events import EventSink
from comfyui_runtime_engine.simulation import EmbeddedSimulation


def test_embedded_simulation(tmp_path: Path) -> None:
    root = Path(__file__).parent / "fixtures" / "fake_comfyui"
    _, inspection = select_adapter(root, ("0.15",), True)
    event_log = tmp_path / "events.jsonl"
    EmbeddedSimulation(inspection, EventSink(event_log)).run()
    text = event_log.read_text(encoding="utf-8")
    assert "simulation.completed" in text
    assert '"cuda_initialized": false' in text
