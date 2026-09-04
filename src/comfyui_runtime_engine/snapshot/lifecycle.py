from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .state import SnapshotStateStore


class SnapshotLifecycle:
    """Provider-neutral snapshot readiness and restoration contract.

    Provider adapters must invoke `before_snapshot` and `after_restore` inside
    the same Python process that owns ComfyUI and CUDA. The local CLI only
    simulates this contract; it does not claim to create a Modal GPU snapshot.
    """

    def __init__(self, state_path: Path, provider: str = "modal") -> None:
        self.store = SnapshotStateStore(state_path)
        self.provider = provider

    def before_snapshot(
        self,
        *,
        residency_guard: dict[str, Any],
        vram: dict[str, Any],
        workflow: str,
        prompt_id: str | None,
        provider_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        manifest = self.store.build_manifest(
            provider=self.provider,
            phase="before_snapshot",
            residents=list(residency_guard.get("protected_loaded_models", [])),
            vram=vram,
            workflow=workflow,
            prompt_id=prompt_id,
        )
        if not manifest["snapshot_ready"]:
            raise RuntimeError(
                "Snapshot blocked: not every configured resident is snapshot-ready (CUDA LoadedModel or retained materialized snapshot object)."
            )
        self.store.write(manifest)
        if provider_callback is not None:
            provider_callback(manifest)
        return manifest

    def after_restore(
        self,
        *,
        residency_guard: dict[str, Any],
        vram: dict[str, Any],
    ) -> dict[str, Any]:
        before = self.store.read()
        current = self.store.build_manifest(
            provider=self.provider,
            phase="after_restore",
            residents=list(residency_guard.get("protected_loaded_models", [])),
            vram=vram,
            workflow=str(before.get("workflow") or ""),
            prompt_id=None,
        )
        expected = {item["identity"] for item in before.get("residents", [])}
        restored = {item["identity"] for item in current.get("residents", [])}
        report = {
            "provider": self.provider,
            "state_path": str(self.store.path),
            "expected_resident_count": len(expected),
            "restored_resident_count": len(restored),
            "missing_identities": sorted(expected - restored),
            "unexpected_identities": sorted(restored - expected),
            "all_expected_residents_present": expected == restored,
            "all_restored_fully_loaded": bool(current["snapshot_ready"]),
            "vram": vram,
            "success": expected == restored and bool(current["snapshot_ready"]),
        }
        return report
