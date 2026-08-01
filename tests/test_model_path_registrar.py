from pathlib import Path
import sys
import types

from comfyui_runtime_engine.warmup import ComfyModelPathRegistrar


def test_registers_standard_and_sam3_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "models"
    for folder in (
        "diffusion_models",
        "text_encoders",
        "vae",
        "sam3",
    ):
        (root / folder).mkdir(parents=True)

    paths: dict[str, list[str]] = {}
    module = types.SimpleNamespace()
    module.filename_list_cache = {"x": object()}
    module.add_model_folder_path = (
        lambda category, path, is_default=False:
        paths.setdefault(category, []).append(path)
    )
    module.get_filename_list = (
        lambda category: ["sample"]
        if category in paths else []
    )
    monkeypatch.setitem(sys.modules, "folder_paths", module)

    report = ComfyModelPathRegistrar((root,)).register()

    assert str(root / "diffusion_models") in paths["diffusion_models"]
    assert str(root / "text_encoders") in paths["text_encoders"]
    assert str(root / "vae") in paths["vae"]
    assert str(root / "sam3") in paths["sam3"]
    assert report["available_counts"]["sam3"] == 1
    assert module.filename_list_cache == {}
