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
        # Some custom loaders (notably JLC Flux2 ControlNet) do not register
        # their materialized side-model in comfy.model_management.current_loaded_models.
        # Keep their returned object graph alive explicitly so Modal can snapshot the
        # already-materialized Python/torch state without weakening the CUDA contract
        # used by normal ComfyUI LoadedModel residents.
        self._external_residents: dict[str, dict[str, Any]] = {}

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

        original_load_clip = getattr(comfy.sd, "load_clip", None)
        if callable(original_load_clip):
            def tagged_load_clip(ckpt_paths, *args, **kwargs):
                clip = original_load_clip(ckpt_paths, *args, **kwargs)
                paths = list(ckpt_paths or []) if isinstance(ckpt_paths, (list, tuple)) else [ckpt_paths]
                resolved = [Path(item).expanduser().resolve() for item in paths if item]
                resident_sources = [path for path in resolved if self._canonical(path) in self.resident_paths]
                patcher = getattr(clip, "patcher", None)
                if patcher is not None and resident_sources:
                    source_path = resident_sources[0]
                    setattr(patcher, "_comfy_runtime_source_path", str(source_path))
                    setattr(patcher, "_comfy_runtime_resident", True)
                    self.events.append({
                        "event": "clip_tagged",
                        "source_path": str(source_path),
                        "resident": True,
                        "patcher_type": type(patcher).__name__,
                    })
                return clip

            comfy.sd.load_clip = tagged_load_clip
            self._restorers.append(
                lambda: setattr(comfy.sd, "load_clip", original_load_clip)
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
            keep_loaded=None,
            for_dynamic=False,
            **kwargs,
        ):
            # ComfyUI 0.31 added pins_required and may add more memory-budget
            # parameters later. Forward them untouched instead of coupling the
            # engine to one exact free_memory() signature.
            keep_loaded = list(keep_loaded or [])
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
                **kwargs,
            )

        model_management.free_memory = guarded_free_memory
        self._restorers.append(
            lambda: setattr(model_management, "free_memory", original_free_memory)
        )

        self._install_external_loader_hooks()
        self._installed = True


    def _install_external_loader_hooks(self) -> None:
        """Capture configured custom-loader objects that live outside LoadedModel.

        JLCFlux2ControlNetLoader intentionally returns a lazy side-model and only
        materializes it later when sampling discovers the base model. Retaining the
        returned object is enough to keep that materialized state reachable for the
        provider snapshot, while the normal five residents remain governed by
        ComfyUI's CUDA LoadedModel machinery.
        """
        try:
            import folder_paths
            import nodes
        except Exception:
            return

        mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
        loader_class = mappings.get("JLCFlux2ControlNetLoader")
        if loader_class is None:
            return
        function_name = getattr(loader_class, "FUNCTION", None)
        if not function_name or not hasattr(loader_class, function_name):
            return
        original = getattr(loader_class, function_name)
        if getattr(original, "_comfy_runtime_external_resident_hook", False):
            return

        try:
            input_types = loader_class.INPUT_TYPES()
            required_names = list((input_types or {}).get("required", {}).keys())
            optional_names = list((input_types or {}).get("optional", {}).keys())
            argument_names = required_names + optional_names
        except Exception:
            argument_names = []

        guard = self

        def wrapped(instance, *args, **kwargs):
            result = original(instance, *args, **kwargs)
            values = dict(kwargs)
            for index, value in enumerate(args):
                if index < len(argument_names):
                    values.setdefault(argument_names[index], value)
            filename = values.get("controlnet_name")
            if filename:
                try:
                    source_path = folder_paths.get_full_path("controlnet", filename)
                except Exception:
                    source_path = None
                if source_path and guard._canonical(source_path) in guard.resident_paths:
                    canonical = guard._canonical(source_path)
                    guard._external_residents[canonical] = {
                        "source_path": str(Path(source_path).expanduser().resolve()),
                        "object": result,
                        "loader": "JLCFlux2ControlNetLoader",
                    }
                    guard.events.append({
                        "event": "external_resident_captured",
                        "source_path": str(Path(source_path).expanduser().resolve()),
                        "loader": "JLCFlux2ControlNetLoader",
                    })
            return result

        setattr(wrapped, "_comfy_runtime_external_resident_hook", True)
        setattr(loader_class, function_name, wrapped)
        self._restorers.append(
            lambda cls=loader_class, name=function_name, method=original:
            setattr(cls, name, method)
        )

    @staticmethod
    def _external_tensor_state(root: Any) -> dict[str, Any]:
        """Summarize reachable torch tensors without copying/moving them."""
        try:
            import torch
        except Exception:
            return {"tensor_bytes": 0, "devices": [], "tensor_count": 0}

        queue = [root]
        seen_objects: set[int] = set()
        seen_tensors: set[int] = set()
        tensor_bytes = 0
        devices: set[str] = set()
        tensor_count = 0
        max_objects = 20000

        while queue and len(seen_objects) < max_objects:
            obj = queue.pop()
            obj_id = id(obj)
            if obj_id in seen_objects:
                continue
            seen_objects.add(obj_id)

            if isinstance(obj, torch.Tensor):
                if obj_id not in seen_tensors:
                    seen_tensors.add(obj_id)
                    tensor_count += 1
                    try:
                        tensor_bytes += int(obj.numel()) * int(obj.element_size())
                    except Exception:
                        pass
                    devices.add(str(obj.device))
                continue
            if isinstance(obj, dict):
                queue.extend(obj.values())
                continue
            if isinstance(obj, (list, tuple, set)):
                queue.extend(obj)
                continue
            if isinstance(obj, (str, bytes, bytearray, int, float, bool, type(None))):
                continue

            values = getattr(obj, "__dict__", None)
            if isinstance(values, dict):
                queue.extend(values.values())

        return {
            "tensor_bytes": tensor_bytes,
            "devices": sorted(devices),
            "tensor_count": tensor_count,
        }

    def external_resident_models(self) -> list[dict[str, Any]]:
        residents: list[dict[str, Any]] = []
        for item in self._external_residents.values():
            source_path = str(item["source_path"])
            state = self._external_tensor_state(item.get("object"))
            tensor_bytes = int(state.get("tensor_bytes") or 0)
            if tensor_bytes <= 0:
                self.events.append({
                    "event": "external_resident_not_materialized",
                    "source_path": source_path,
                    "loader": item.get("loader"),
                })
                continue
            residents.append({
                "source_path": source_path,
                "model_type": item.get("loader") or "external_loader",
                "device": "snapshot-object",
                "currently_used": True,
                "loaded_bytes": tensor_bytes,
                "total_model_bytes": tensor_bytes,
                "residency_mode": "snapshot_object",
                "tensor_devices": state.get("devices", []),
                "tensor_count": int(state.get("tensor_count") or 0),
            })
        return residents

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
        external = self.external_resident_models()
        normal = [
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
            ]
        combined = normal + external
        return {
            "configured_resident_paths": sorted(self.resident_paths),
            "protected_loaded_count": len(combined),
            "protected_loaded_models": combined,
            "loadedmodel_resident_count": len(normal),
            "external_resident_count": len(external),
            "external_resident_models": external,
            "guard_events": list(self.events),
        }
