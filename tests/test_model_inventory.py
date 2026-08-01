from pathlib import Path

from comfyui_runtime_engine.residency import ModelInventory


def test_inventory_scans_and_resolves_unique_filename(tmp_path: Path) -> None:
    root = tmp_path / "models"
    model = root / "diffusion_models" / "heavy.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"1234")

    inventory = ModelInventory((root,))
    items = inventory.scan()
    assert len(items) == 1
    assert items[0].inferred_kind == "diffusion_model"

    resolution = inventory.resolve(("heavy.safetensors",), items)[0]
    assert resolution.resolved is True
    assert resolution.matches[0].size_bytes == 4


def test_inventory_marks_duplicate_filenames_ambiguous(tmp_path: Path) -> None:
    root = tmp_path / "models"
    for folder in ("loras/a", "loras/b"):
        path = root / folder / "same.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")

    result = ModelInventory((root,)).resolve(("same.safetensors",))[0]
    assert result.status == "ambiguous"
    assert len(result.matches) == 2
