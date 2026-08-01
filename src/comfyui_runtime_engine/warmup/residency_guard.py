from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable


class SelectiveResidencyGuard(AbstractContextManager):
    """Protect tagged resident diffusion models from standard ComfyUI purges.

    The guard does not disable memory management globally. It augments
    model_management.free_memory(..., keep_loaded=...) only with LoadedModel
    objects whose ModelPatcher was tagged from a configured resident file.
    """

    def __init__(self, resident_paths: tuple[Path, ...]) -> None:
        self.resident_paths = {
            self._canonical(path) for path in resident_paths
        }
        self.events: list[dict[str, Any]] = []
        self._restorers: list[Callable[[], None]] = []
        self._installed = False

    def __enter__(self) -> "SelectiveResidencyGuard":
        self.install()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.restore()

    @staticmethod
    def _canonical(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve()).casefold()

    def install(self) -> None:
        if self._installed:
            return

        import comfy.sd
        import comfy.model_management as model_management
        import comfy.model_patcher as model_patcher

        original_load_diffusion_model = comfy.sd.load_diffusion_model

        def tagged_load_diffusion_model(
            unet_path,
            model_options={},
            disable_dynamic=False,
        ):
            patcher = original_load_diffusion_model(
                unet_path,
                model_options=model_options,
                disable_dynamic=disable_dynamic,
            )
            source = self._canonical(unet_path)
            setattr(patcher, "_comfy_runtime_source_path", str(Path(unet_path).resolve()))
            setattr(patcher, "_comfy_runtime_resident", source in self.resident_paths)
            self.events.append(
                {
                    "event": "model_tagged",
                    "source_path": str(Path(unet_path).resolve()),
                    "resident": source in self.resident_paths,
                    "patcher_type": type(patcher).__name__,
                }
            )
            return patcher

        comfy.sd.load_diffusion_model = tagged_load_diffusion_model
        self._restorers.append(
            lambda: setattr(comfy.sd, "load_diffusion_model", original_load_diffusion_model)
        )

        for patcher_class_name in ("ModelPatcher", "CoreModelPatcher"):
            patcher_class = getattr(model_patcher, patcher_class_name, None)
            if patcher_class is None or not hasattr(patcher_class, "clone"):
                continue
            original_clone = patcher_class.clone

            def make_clone_wrapper(original):
                def clone_wrapper(instance, *args, **kwargs):
                    cloned = original(instance, *args, **kwargs)
                    for attr in (
                        "_comfy_runtime_source_path",
                        "_comfy_runtime_resident",
                    ):
                        if hasattr(instance, attr):
                            setattr(cloned, attr, getattr(instance, attr))
                    return cloned
                return clone_wrapper

            patcher_class.clone = make_clone_wrapper(original_clone)
            self._restorers.append(
                lambda cls=patcher_class, method=original_clone:
                setattr(cls, "clone", method)
            )

        original_free_memory = model_management.free_memory

        def guarded_free_memory(
            memory_required,
            device,
            keep_loaded=[],
            for_dynamic=False,
            ram_required=0,
        ):
            protected = self.protected_loaded_models(model_management)
            merged = list(keep_loaded)
            for loaded in protected:
                if loaded not in merged:
                    merged.append(loaded)
            self.events.append(
                {
                    "event": "free_memory",
                    "memory_required": float(memory_required),
                    "device": str(device),
                    "caller_keep_loaded": len(keep_loaded),
                    "protected_added": len(protected),
                    "protected_sources": [
                        getattr(loaded.model, "_comfy_runtime_source_path", None)
                        for loaded in protected
                    ],
                }
            )
            return original_free_memory(
                memory_required,
                device,
                keep_loaded=merged,
                for_dynamic=for_dynamic,
                ram_required=ram_required,
            )

        model_management.free_memory = guarded_free_memory
        self._restorers.append(
            lambda: setattr(model_management, "free_memory", original_free_memory)
        )

        self._installed = True

    def restore(self) -> None:
        for restore in reversed(self._restorers):
            restore()
        self._restorers.clear()
        self._installed = False

    def protected_loaded_models(self, model_management=None) -> list[Any]:
        if model_management is None:
            import comfy.model_management as model_management
        protected = []
        for loaded in list(model_management.current_loaded_models):
            patcher = getattr(loaded, "model", None)
            if patcher is None:
                continue
            source = getattr(patcher, "_comfy_runtime_source_path", None)
            explicit = bool(getattr(patcher, "_comfy_runtime_resident", False))
            if explicit or (
                source is not None and self._canonical(source) in self.resident_paths
            ):
                protected.append(loaded)
        return protected

    def report(self) -> dict[str, Any]:
        import comfy.model_management as model_management

        protected = self.protected_loaded_models(model_management)
        return {
            "configured_resident_paths": sorted(self.resident_paths),
            "protected_loaded_count": len(protected),
            "protected_loaded_models": [
                {
                    "source_path": getattr(
                        loaded.model, "_comfy_runtime_source_path", None
                    ),
                    "model_type": type(loaded.model).__name__,
                    "device": str(getattr(loaded, "device", "")),
                    "currently_used": bool(
                        getattr(loaded, "currently_used", False)
                    ),
                    "loaded_bytes": int(loaded.model_loaded_memory()),
                    "total_model_bytes": int(loaded.model_memory()),
                }
                for loaded in protected
            ],
            "guard_events": list(self.events),
        }
