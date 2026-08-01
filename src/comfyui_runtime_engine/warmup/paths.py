from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RegisteredModelPath:
    category: str
    path: str
    exists: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ComfyModelPathRegistrar:
    """Register external model roots in ComfyUI's real folder_paths registry."""

    CATEGORY_SUBDIRS: dict[str, tuple[str, ...]] = {
        "checkpoints": ("checkpoints",),
        "configs": ("configs",),
        "loras": ("loras",),
        "vae": ("vae",),
        "text_encoders": ("text_encoders", "clip"),
        "diffusion_models": ("diffusion_models", "unet"),
        "clip_vision": ("clip_vision",),
        "style_models": ("style_models",),
        "embeddings": ("embeddings",),
        "diffusers": ("diffusers",),
        "vae_approx": ("vae_approx",),
        "controlnet": ("controlnet", "t2i_adapter"),
        "gligen": ("gligen",),
        "upscale_models": ("upscale_models",),
        "hypernetworks": ("hypernetworks",),
        "photomaker": ("photomaker",),
        # Custom category used by the installed TBG SAM3 node.
        "sam3": ("sam3",),
    }

    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = tuple(path.expanduser().resolve() for path in roots)

    def register(self) -> dict[str, Any]:
        import folder_paths

        registered: list[RegisteredModelPath] = []
        for root in self.roots:
            for category, subdirs in self.CATEGORY_SUBDIRS.items():
                for subdir in subdirs:
                    candidate = (root / subdir).resolve()
                    exists = candidate.is_dir()
                    if exists:
                        folder_paths.add_model_folder_path(
                            category,
                            str(candidate),
                            is_default=False,
                        )
                    registered.append(
                        RegisteredModelPath(category, str(candidate), exists)
                    )

        # Invalidate ComfyUI filename caches after adding paths.
        cache = getattr(folder_paths, "filename_list_cache", None)
        if isinstance(cache, dict):
            cache.clear()

        available: dict[str, list[str]] = {}
        for category in self.CATEGORY_SUBDIRS:
            try:
                available[category] = list(folder_paths.get_filename_list(category))
            except Exception:
                available[category] = []

        return {
            "roots": [str(root) for root in self.roots],
            "registered": [
                item.to_dict() for item in registered if item.exists
            ],
            "missing_directories": [
                item.to_dict() for item in registered if not item.exists
            ],
            "available_counts": {
                category: len(names) for category, names in available.items()
            },
            "available_samples": {
                category: names[:20] for category, names in available.items()
            },
        }
