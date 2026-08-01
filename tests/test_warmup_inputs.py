from pathlib import Path

from comfyui_runtime_engine.warmup import (
    WarmupInput,
    WarmupInputStager,
)


def test_stages_images_using_exact_workflow_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    input_dir = tmp_path / "input"

    report = WarmupInputStager(input_dir).stage(
        (
            WarmupInput(
                source=source,
                target="workflow-image.png",
            ),
        )
    )

    assert report["count"] == 1
    assert (
        input_dir / "workflow-image.png"
    ).read_bytes() == b"image"
