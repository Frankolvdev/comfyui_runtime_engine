from comfyui_runtime_engine.events import EventSink
from comfyui_runtime_engine.gpu import CUDAEnvironmentSetup, CUDAInstallPlan


def test_cuda_setup_dry_run_builds_coordinated_cu128_plan(tmp_path) -> None:
    report = CUDAEnvironmentSetup(
        EventSink(tmp_path / "events.jsonl")
    ).prepare(dry_run=True, channel="cu128")
    assert report["success"] is True
    install = report["commands"][1]
    assert "torch==2.11.0" in install
    assert "torchvision==0.26.0" in install
    assert "torchaudio==2.11.0" in install
    assert install[-1] == "https://download.pytorch.org/whl/cu128"


def test_cuda_plan_accepts_cu130() -> None:
    plan = CUDAInstallPlan(channel="cu130")
    assert plan.index_url.endswith("/cu130")
