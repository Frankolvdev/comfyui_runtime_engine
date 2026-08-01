import os

from comfyui_runtime_engine.embedded import local_probe_environment


def test_local_probe_environment_is_reversible() -> None:
    os.environ.pop("COMFY_RUNTIME_OFFLINE", None)
    with local_probe_environment(True):
        assert os.environ["COMFY_RUNTIME_OFFLINE"] == "1"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "COMFY_RUNTIME_OFFLINE" not in os.environ
