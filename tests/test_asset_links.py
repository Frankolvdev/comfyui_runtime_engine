from pathlib import Path

from comfyui_runtime_engine.warmup import AssetLinkManager


def test_asset_link_uses_hardlink_when_requested(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    expected = tmp_path / "runtime" / "sam3.pt"
    source.write_bytes(b"weights")

    result = AssetLinkManager().ensure_file_link(
        name="sam3.pt",
        source=source,
        expected=expected,
        mode="hardlink",
    )

    assert result.status == "created"
    assert result.link_type == "hardlink"
    assert expected.read_bytes() == b"weights"
