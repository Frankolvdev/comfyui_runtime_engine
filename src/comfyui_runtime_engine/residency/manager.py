from __future__ import annotations

from pathlib import Path
from typing import Any

from ..events import EventSink
from .models import ModelInventory


class ResidencyManager:
    """Build a deterministic model-residency plan without loading models yet."""

    def __init__(
        self,
        *,
        comfyui_root: Path,
        model_roots: tuple[Path, ...],
        resident_models: tuple[str, ...],
        events: EventSink,
    ) -> None:
        roots = model_roots or (comfyui_root / "models",)
        self.inventory = ModelInventory(roots)
        self.resident_models = resident_models
        self.events = events

    def scan(self) -> dict[str, Any]:
        items = self.inventory.scan()
        total_bytes = sum(item.size_bytes for item in items)
        by_kind: dict[str, dict[str, int]] = {}
        for item in items:
            bucket = by_kind.setdefault(item.inferred_kind, {"count": 0, "size_bytes": 0})
            bucket["count"] += 1
            bucket["size_bytes"] += item.size_bytes

        report = {
            "roots": [str(root) for root in self.inventory.roots],
            "count": len(items),
            "total_bytes": total_bytes,
            "by_kind": by_kind,
            "models": [item.to_dict() for item in items],
        }
        self.events.emit(
            "residency.scan.completed",
            roots=report["roots"],
            count=report["count"],
            total_bytes=report["total_bytes"],
            by_kind=report["by_kind"],
        )
        return report

    def plan(self) -> dict[str, Any]:
        items = self.inventory.scan()
        resolutions = self.inventory.resolve(self.resident_models, items)
        resolved = [item for item in resolutions if item.resolved]
        missing = [item.requested for item in resolutions if item.status == "missing"]
        ambiguous = [item.requested for item in resolutions if item.status == "ambiguous"]
        estimated_weight_bytes = sum(
            resolution.matches[0].size_bytes for resolution in resolved
        )
        required_ok = not missing and not ambiguous and bool(self.resident_models)
        report = {
            "configured": list(self.resident_models),
            "roots": [str(root) for root in self.inventory.roots],
            "resolutions": [item.to_dict() for item in resolutions],
            "resolved_count": len(resolved),
            "estimated_weight_bytes": estimated_weight_bytes,
            "missing": missing,
            "ambiguous": ambiguous,
            "ready_for_warmup": required_ok,
            "load_strategy": "workflow-driven",
            "note": (
                "The engine resolves files now. Actual model loading will be performed "
                "by the matching ComfyUI loader nodes through a warmup workflow."
            ),
        }
        self.events.emit("residency.plan.completed", **report)
        return report

    def verify(self) -> dict[str, Any]:
        report = self.plan()
        report["success"] = bool(report["ready_for_warmup"])
        self.events.emit(
            "residency.verify.completed",
            success=report["success"],
            missing=report["missing"],
            ambiguous=report["ambiguous"],
        )
        return report
