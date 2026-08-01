from pathlib import Path

from comfyui_runtime_engine.warmup.residency_guard import SelectiveResidencyGuard


def test_guard_canonicalizes_resident_paths(tmp_path: Path) -> None:
    model = tmp_path / "models" / "resident.safetensors"
    model.parent.mkdir()
    model.write_bytes(b"x")

    guard = SelectiveResidencyGuard((model,))

    assert str(model.resolve()).casefold() in guard.resident_paths
