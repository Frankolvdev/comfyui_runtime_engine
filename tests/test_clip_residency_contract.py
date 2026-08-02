from pathlib import Path
from comfyui_runtime_engine.warmup.residency_guard import SelectiveResidencyGuard


def test_clip_resident_path_is_canonicalized(tmp_path: Path):
    model = tmp_path / "text_encoders" / "qwen.safetensors"
    model.parent.mkdir(); model.write_bytes(b"x")
    guard = SelectiveResidencyGuard((model,))
    assert guard._canonical(model) in guard.resident_paths
