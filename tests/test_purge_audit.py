import json
from pathlib import Path

from comfyui_runtime_engine.warmup import WarmupWorkflow


def test_detects_purge_nodes_without_modifying_workflow(tmp_path: Path) -> None:
    path = tmp_path / "workflow.json"
    original = {
        "1": {
            "class_type": "PurgeVRAM",
            "inputs": {"anything": True},
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "model.safetensors"},
        },
    }
    path.write_text(json.dumps(original), encoding="utf-8")
    workflow = WarmupWorkflow.load(path)

    matches = workflow.purge_nodes()

    assert len(matches) == 1
    assert matches[0]["node_id"] == "1"
    assert workflow.prompt == original
