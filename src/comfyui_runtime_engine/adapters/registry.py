from __future__ import annotations

from pathlib import Path

from ..errors import CompatibilityError
from .base import ComfyAdapter, ComfyInspection
from .comfy_v015 import ComfyV015Adapter


def select_adapter(root: Path, supported_versions: tuple[str, ...], strict: bool) -> tuple[ComfyAdapter, ComfyInspection]:
    candidate = ComfyV015Adapter()
    inspection = candidate.inspect(root)

    if not inspection.structurally_valid:
        missing = ", ".join(str(path.relative_to(root)) for path in inspection.missing_paths)
        raise CompatibilityError(f"Invalid ComfyUI root. Missing: {missing}")

    version_supported = inspection.version == "unknown" or any(
        inspection.version.startswith(prefix) for prefix in supported_versions
    )
    if strict and not version_supported:
        raise CompatibilityError(
            f"ComfyUI {inspection.version} is not supported by this engine build. "
            f"Configured families: {supported_versions}"
        )
    return candidate, inspection
