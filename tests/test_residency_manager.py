from pathlib import Path

from comfyui_runtime_engine.events import EventSink
from comfyui_runtime_engine.residency import ResidencyManager


def test_residency_plan_requires_all_models(tmp_path: Path) -> None:
    models = tmp_path / "models"
    path = models / "vae" / "vae.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"weights")

    manager = ResidencyManager(
        comfyui_root=tmp_path,
        model_roots=(models,),
        resident_models=("vae/vae.safetensors",),
        events=EventSink(tmp_path / "events.jsonl"),
    )
    report = manager.plan()
    assert report["ready_for_warmup"] is True
    assert report["resolved_count"] == 1


def test_residency_plan_reports_missing(tmp_path: Path) -> None:
    manager = ResidencyManager(
        comfyui_root=tmp_path,
        model_roots=(tmp_path / "models",),
        resident_models=("missing.safetensors",),
        events=EventSink(tmp_path / "events.jsonl"),
    )
    report = manager.verify()
    assert report["success"] is False
    assert report["missing"] == ["missing.safetensors"]
