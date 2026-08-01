import json
from pathlib import Path
import pytest
from comfyui_runtime_engine.warmup import WarmupWorkflow

def test_loads_api_workflow(tmp_path: Path):
    p=tmp_path/"w.json"
    p.write_text(json.dumps({"1":{"class_type":"UNETLoader","inputs":{"unet_name":"x.safetensors"}}}))
    w=WarmupWorkflow.load(p)
    assert w.node_count==1

def test_rejects_ui_workflow(tmp_path: Path):
    p=tmp_path/"w.json"; p.write_text(json.dumps({"nodes":[]}))
    with pytest.raises(ValueError): WarmupWorkflow.load(p)
