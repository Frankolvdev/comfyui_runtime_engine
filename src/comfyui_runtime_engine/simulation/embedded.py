from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ..adapters.base import ComfyInspection
from ..events import EventSink
from ..lifecycle import LifecyclePhase, LifecycleState


class EmbeddedSimulation:
    """Local lifecycle simulation that deliberately avoids importing CUDA/ComfyUI."""

    def __init__(self, inspection: ComfyInspection, events: EventSink) -> None:
        self._inspection = inspection
        self._events = events
        self._state = LifecycleState()

    def run(self) -> None:
        self._events.emit("simulation.created", root=str(self._inspection.root))
        self._state.transition(LifecyclePhase.INSPECTED)
        self._events.emit(
            "simulation.inspected",
            version=self._inspection.version,
            main_file=str(self._inspection.main_file),
        )

        self._state.transition(LifecyclePhase.INITIALIZING)
        self._events.emit("simulation.initializing", strategy="embedded-no-import")
        self._validate_import_visibility(self._inspection.root)

        self._state.transition(LifecyclePhase.READY)
        self._events.emit(
            "simulation.ready",
            phase=self._state.phase.value,
            cuda_initialized=False,
            comfyui_imported=False,
        )

        self._state.transition(LifecyclePhase.STOPPING)
        self._events.emit("simulation.stopping")
        self._state.transition(LifecyclePhase.STOPPED)
        self._events.emit("simulation.completed", phase=self._state.phase.value)

    @staticmethod
    def _validate_import_visibility(root: Path) -> None:
        original_path = list(sys.path)
        try:
            sys.path.insert(0, str(root))
            for module_name in ("server", "execution", "nodes", "folder_paths"):
                spec = importlib.util.find_spec(module_name)
                if spec is None or spec.origin is None:
                    raise RuntimeError(f"Module is not import-visible: {module_name}")
        finally:
            sys.path[:] = original_path
