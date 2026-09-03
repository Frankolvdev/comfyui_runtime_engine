
from pathlib import Path


def test_config_has_python310_tomli_fallback():
    source = Path("src/comfyui_runtime_engine/config.py").read_text(encoding="utf-8")
    assert "import tomllib" in source
    assert "except ModuleNotFoundError" in source
    assert "import tomli as tomllib" in source


def test_pyproject_installs_tomli_only_for_python_below_311():
    source = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'tomli>=2.0.1; python_version < \\"3.11\\"' in source
