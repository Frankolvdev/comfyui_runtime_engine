from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import time
from typing import Any

from ..embedded import EmbeddedWarmupRunner
from ..errors import RuntimeEngineError
from ..runtime import RuntimeEngine
from ..snapshot import SnapshotLifecycle
from ..warmup import AssetLinkManager, WarmupWorkflow
from .contracts import AtomicJsonFile


class PersistentModalRuntime:
    """Run one embedded ComfyUI process through the Modal snapshot lifecycle."""

    def __init__(
        self,
        engine: RuntimeEngine,
        *,
        ready_path: Path,
    ) -> None:
        self.engine = engine
        self.config = engine.config
        self.ready_store = AtomicJsonFile(ready_path)
        self._stopping = False

    def serve(self) -> int:
        if not self.config.snapshot_enabled:
            raise RuntimeEngineError(
                "snapshot.enabled must be true for Modal persistent mode"
            )
        if not self.config.gpu_snapshot_enabled:
            raise RuntimeEngineError(
                "snapshot.gpu_enabled must be true for Modal persistent mode"
            )
        verification = self.engine.warmup_verify()
        if not verification["success"]:
            raise RuntimeEngineError(
                "Warmup verification failed before Modal startup: "
                + json.dumps(verification, ensure_ascii=False)
            )

        # SAM3 used to be mandatory because the original snapshot profile was
        # SAM3-centric. Flux-only snapshots must not pay for or require it.
        sam3_selected = any(
            str(item).strip().replace("\\", "/").casefold().startswith("sam3/")
            for item in self.config.resident_models
        )
        asset_links: dict[str, Any] = {}
        if sam3_selected:
            asset_result = AssetLinkManager(
                replace_existing=self.config.sam3_replace_existing
            ).ensure_file_link(
                name="sam3.pt",
                source=self.config.sam3_source_model,
                expected=self.config.sam3_expected_model,
                mode=self.config.sam3_link_mode,
            )
            if asset_result.status in {
                "source_missing",
                "existing_target_not_replaced",
            }:
                raise RuntimeEngineError(
                    "SAM3 asset preparation is not ready: "
                    + json.dumps(asset_result.to_dict(), ensure_ascii=False)
                )
            asset_links["sam3"] = asset_result.to_dict()

        resident_plan = self.engine.residency_plan()
        resident_paths = tuple(
            Path(item["matches"][0]["absolute_path"])
            for item in resident_plan["resolutions"]
            if item["resolved"]
        )
        gpu_args_list = [
            argument
            for argument in self.config.embedded_extra_args
            if argument != "--cpu"
        ]
        # The normal TRYON Modal runtime disables DynamicVRAM because it caused
        # severe loading stalls. The embedded snapshot bootstrap must use the same
        # memory policy. Without this flag ComfyUI 0.31 leaves Flux2/CLIP as
        # dynamic-VRAM "Staged" models, so a GPU snapshot can never satisfy the
        # full-residency contract even after a force_full_load promotion.
        if "--disable-dynamic-vram" not in gpu_args_list:
            gpu_args_list.append("--disable-dynamic-vram")
        gpu_args = tuple(gpu_args_list)
        lifecycle = SnapshotLifecycle(
            self.config.snapshot_state_path,
            provider="modal",
        )

        runner = EmbeddedWarmupRunner(
            self.engine._embedded_bootstrap(extra_args=gpu_args),
            model_roots=self.engine._effective_model_roots(),
            warmup_inputs=self.config.warmup_inputs,
            resident_paths=resident_paths,
        )
        workflow = WarmupWorkflow.load(self.config.warmup_workflow)

        report = runner.run(
            workflow,
            timeout_seconds=self.config.warmup_timeout_seconds,
            hold_seconds=0,
            iterations=1,
            asset_links=asset_links,
            snapshot_lifecycle=lifecycle,
            snapshot_audit=None,
            persist_after_ready=True,
            persistent_ready_callback=self._publish_ready,
        )
        # Persistent mode only returns when the runtime is stopped.
        return 0 if report.get("success") else 1

    def _publish_ready(self, report: dict[str, Any]) -> None:
        snapshot_state = report.get("snapshot_state") or {}
        residents = list(snapshot_state.get("residents", []))
        payload = {
            "schema_version": 1,
            "phase": "before_snapshot",
            "pid": os.getpid(),
            "snapshot_ready": bool(snapshot_state.get("snapshot_ready")),
            "resident_count": len(residents),
            "expected_resident_count": len(self.config.resident_models),
            "resident_identities": [
                {
                    "identity": item.get("identity"),
                    "source_path": item.get("source_path"),
                    "device": item.get("device"),
                    "loaded_bytes": item.get("loaded_bytes"),
                    "total_model_bytes": item.get("total_model_bytes"),
                    "fully_loaded": item.get("fully_loaded"),
                }
                for item in residents
            ],
            "vram": snapshot_state.get("vram", {}),
            "workflow": snapshot_state.get("workflow"),
            "prompt_id": snapshot_state.get("prompt_id"),
            "created_at": snapshot_state.get("created_at"),
        }
        self.ready_store.write(payload)
