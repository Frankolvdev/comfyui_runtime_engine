from pathlib import Path

from comfyui_runtime_engine.adapters import select_adapter


def test_select_v015_adapter() -> None:
    root = Path(__file__).parent / "fixtures" / "fake_comfyui"
    adapter, inspection = select_adapter(root, ("0.15",), True)
    assert adapter.family == "0.15"
    assert inspection.version == "0.15.1"
    assert inspection.structurally_valid
