from pathlib import Path

from comfyui_runtime_engine.config import RuntimeConfig


def test_load_config(tmp_path: Path) -> None:
    comfy = Path(__file__).parent / "fixtures" / "fake_comfyui"
    config_path = tmp_path / "engine.toml"
    config_path.write_text(
        f'''[runtime]\nmode = "embedded-simulation"\ncomfyui_path = "{comfy.as_posix()}"\n\n[embedded]\nshutdown_grace_seconds = 3.5\nensure_workspace = true\n''',
        encoding="utf-8",
    )
    config = RuntimeConfig.from_toml(config_path)
    assert config.mode == "embedded-simulation"
    assert config.comfyui_path == comfy.resolve()
    assert config.shutdown_grace_seconds == 3.5
    assert config.ensure_workspace is True
