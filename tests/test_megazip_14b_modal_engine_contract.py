from app.services.runtime_builder_service import RuntimeBuilderService


def test_modal_runtime_engine_config_is_selective():
    content = RuntimeBuilderService._modal_runtime_engine_toml("/models")
    assert 'gpu_enabled = true' in content
    assert '"diffusion_models/realDream_klein9BV1.safetensors"' in content
    assert '"text_encoders/qwen_3_8b.safetensors"' in content
    assert 'model_roots = ["/models"]' in content
    assert 'warmup_workflow = "/app/runtime/modal-snapshot-warmup.json"' in content


def test_modal_warmup_only_loads_resident_model():
    content = RuntimeBuilderService._modal_snapshot_warmup_workflow()
    assert "realDream_klein9BV1.safetensors" in content
    assert "TryonSnapshotWarmupSink" in content
    assert "qwen_3_8b.safetensors" in content
    assert "flux2-vae.safetensors" not in content
    assert "sam3.pt" not in content


def test_modal_app_keeps_pipeline_and_adds_engine_hooks():
    content = RuntimeBuilderService._modal_app("test-models", "/models", "test-runtime")
    assert "def run_pipeline(self, payload):" in content
    assert "from generation_runtime import GenerationRuntime" in content
    assert "ModalSnapshotAdapter" in content
    assert "snapshot_adapter.prepare_snapshot()" in content
    assert "adapter.after_restore()" in content
    assert "TRYON_MODAL_RUNTIME_ENGINE_ENABLED" in content
