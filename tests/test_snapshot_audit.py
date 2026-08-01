from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from comfyui_runtime_engine.snapshot import SnapshotCompatibilityAudit


def test_audit_marks_network_resources_for_recreation(tmp_path: Path) -> None:
    loop = asyncio.new_event_loop()
    try:
        prompt_server = SimpleNamespace(runner=None, site=None)
        prompt_queue = SimpleNamespace(queue=[], history={})
        audit = SnapshotCompatibilityAudit(tmp_path / "audit.json")
        report = audit.run(
            loop=loop,
            prompt_server=prompt_server,
            prompt_queue=prompt_queue,
            residency_guard={
                "protected_loaded_models": [
                    {
                        "source_path": str(tmp_path / "model.safetensors"),
                        "device": "cuda:0",
                        "loaded_bytes": 100,
                        "total_model_bytes": 100,
                    }
                ]
            },
            snapshot_state={"snapshot_ready": True},
            model_paths={"registered": [{"category": "diffusion_models"}]},
        )
    finally:
        loop.close()

    findings = {
        item["component"]: item for item in report["findings"]
    }
    assert (
        findings["prompt_server_network"]["disposition"]
        == "recreate_after_restore"
    )
    assert findings["cuda_resident_models"]["severity"] == "info"
    assert report["summary"]["snapshot_compatible"] is True
    assert (tmp_path / "audit.json").is_file()


def test_audit_blocks_missing_cuda_residents(tmp_path: Path) -> None:
    loop = asyncio.new_event_loop()
    try:
        report = SnapshotCompatibilityAudit(
            tmp_path / "audit.json"
        ).run(
            loop=loop,
            prompt_server=SimpleNamespace(runner=None, site=None),
            prompt_queue=SimpleNamespace(queue=[], history={}),
            residency_guard={"protected_loaded_models": []},
            snapshot_state={"snapshot_ready": False},
            model_paths={"registered": []},
        )
    finally:
        loop.close()

    assert report["summary"]["snapshot_compatible"] is False
    assert "cuda_resident_models" in report["summary"]["blockers"]
