from pathlib import Path

from comfyui_runtime_engine.snapshot import SnapshotLifecycle, SnapshotStateStore


def _resident(path: Path):
    return {
        "source_path": str(path),
        "loaded_bytes": 100,
        "total_model_bytes": 100,
        "device": "cuda:0",
    }


def test_snapshot_manifest_requires_fully_loaded_cuda_model(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"x")
    store = SnapshotStateStore(tmp_path / "state.json")
    manifest = store.build_manifest(
        provider="local-simulation",
        phase="before_snapshot",
        residents=[_resident(model)],
        vram={"memory_allocated_bytes": 100},
        workflow="warmup.json",
        prompt_id="abc",
    )
    assert manifest["snapshot_ready"] is True
    assert manifest["resident_count"] == 1


def test_restore_compares_model_identity(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"x")
    lifecycle = SnapshotLifecycle(tmp_path / "state.json", "local-simulation")
    before = lifecycle.before_snapshot(
        residency_guard={"protected_loaded_models": [_resident(model)]},
        vram={"memory_allocated_bytes": 100},
        workflow="warmup.json",
        prompt_id="abc",
    )
    report = lifecycle.after_restore(
        residency_guard={"protected_loaded_models": [_resident(model)]},
        vram={"memory_allocated_bytes": 100},
    )
    assert before["snapshot_ready"] is True
    assert report["success"] is True
