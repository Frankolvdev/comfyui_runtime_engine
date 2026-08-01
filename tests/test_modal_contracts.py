from pathlib import Path

from comfyui_runtime_engine.modal import AtomicJsonFile


def test_atomic_json_roundtrip(tmp_path: Path) -> None:
    store = AtomicJsonFile(tmp_path / "state.json")
    store.write({"snapshot_ready": True})
    assert store.read() == {"snapshot_ready": True}
