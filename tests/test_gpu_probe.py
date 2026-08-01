from comfyui_runtime_engine.events import EventSink
from comfyui_runtime_engine.gpu import GPUProbe


def test_gpu_probe_returns_structured_report(tmp_path) -> None:
    report = GPUProbe(EventSink(tmp_path / "events.jsonl")).run()
    assert "torch_imported" in report
    assert "cuda_available" in report
    assert "devices" in report
    assert "nvidia_smi" in report
    assert isinstance(report["success"], bool)
