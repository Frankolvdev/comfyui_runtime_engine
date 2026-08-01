from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

MODEL_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".onnx",
}


@dataclass(frozen=True, slots=True)
class ModelFile:
    root: str
    absolute_path: str
    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    inferred_kind: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelResolution:
    requested: str
    status: str
    matches: tuple[ModelFile, ...]

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and len(self.matches) == 1

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "status": self.status,
            "resolved": self.resolved,
            "matches": [item.to_dict() for item in self.matches],
        }


def infer_model_kind(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    rules = (
        ({"checkpoints", "checkpoint"}, "checkpoint"),
        ({"diffusion_models", "unet", "unets"}, "diffusion_model"),
        ({"text_encoders", "clip"}, "text_encoder"),
        ({"vae", "vae_approx"}, "vae"),
        ({"loras", "lora"}, "lora"),
        ({"controlnet", "controlnets"}, "controlnet"),
        ({"embeddings"}, "embedding"),
        ({"upscale_models"}, "upscale_model"),
        ({"sam", "sam2", "sam3"}, "segmentation_model"),
    )
    for names, kind in rules:
        if parts.intersection(names):
            return kind
    if path.suffix.lower() == ".gguf":
        return "gguf"
    return "unknown"


class ModelInventory:
    def __init__(self, roots: Iterable[Path]) -> None:
        unique: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            resolved = root.expanduser().resolve()
            key = str(resolved).casefold()
            if key not in seen:
                seen.add(key)
                unique.append(resolved)
        self.roots = tuple(unique)

    def scan(self) -> tuple[ModelFile, ...]:
        items: list[ModelFile] = []
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
                    continue
                try:
                    relative = path.relative_to(root)
                    size = path.stat().st_size
                except OSError:
                    continue
                items.append(
                    ModelFile(
                        root=str(root),
                        absolute_path=str(path.resolve()),
                        relative_path=relative.as_posix(),
                        filename=path.name,
                        extension=path.suffix.lower(),
                        size_bytes=size,
                        inferred_kind=infer_model_kind(relative),
                    )
                )
        return tuple(sorted(items, key=lambda item: item.relative_path.casefold()))

    def resolve(
        self,
        requested: Iterable[str],
        inventory: tuple[ModelFile, ...] | None = None,
    ) -> tuple[ModelResolution, ...]:
        items = inventory if inventory is not None else self.scan()
        results: list[ModelResolution] = []

        for raw_name in requested:
            name = raw_name.strip().replace("\\", "/")
            key = name.casefold()
            exact_relative = tuple(
                item for item in items if item.relative_path.casefold() == key
            )
            if len(exact_relative) == 1:
                results.append(ModelResolution(raw_name, "resolved", exact_relative))
                continue
            if len(exact_relative) > 1:
                results.append(ModelResolution(raw_name, "ambiguous", exact_relative))
                continue

            exact_filename = tuple(
                item for item in items if item.filename.casefold() == key
            )
            if len(exact_filename) == 1:
                results.append(ModelResolution(raw_name, "resolved", exact_filename))
            elif len(exact_filename) > 1:
                results.append(ModelResolution(raw_name, "ambiguous", exact_filename))
            else:
                results.append(ModelResolution(raw_name, "missing", ()))

        return tuple(results)
