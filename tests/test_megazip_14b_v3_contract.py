from app.services.runtime_builder_service import RuntimeBuilderService


def test_modal_engine_config_and_selective_warmup():
    config = RuntimeBuilderService._modal_runtime_engine_toml("/models")
    warmup = RuntimeBuilderService._modal_snapshot_warmup_workflow()

    assert 'gpu_enabled = true' in config
    assert '"diffusion_models/realDream_klein9BV1.safetensors"' in config
    assert '"text_encoders/qwen_3_8b.safetensors"' in config
    assert 'model_roots = ["/models"]' in config
    assert "realDream_klein9BV1.safetensors" in warmup
    assert "qwen_3_8b.safetensors" in warmup
    assert "flux2-vae.safetensors" not in warmup


def test_modal_app_preserves_pipeline_and_adds_engine():
    app = RuntimeBuilderService._modal_app(
        "test-models",
        "/models",
        "test-runtime",
    )

    assert "def run_pipeline(self, payload):" in app
    assert "from generation_runtime import GenerationRuntime" in app
    assert "ModalSnapshotAdapter" in app
    assert "snapshot_adapter.prepare_snapshot()" in app
    assert "adapter.after_restore()" in app
    assert "TRYON_MODAL_RUNTIME_ENGINE_ENABLED" in app


def test_modal_engine_import_is_deferred_to_modal_container():
    app = RuntimeBuilderService._modal_app(
        "test-models",
        "/models",
        "test-runtime",
    )

    top_level_prefix = app.split("APP_NAME =", 1)[0]
    assert "from comfyui_runtime_engine.modal import ModalSnapshotAdapter" not in top_level_prefix

    snapshot_block = app.split("def initialize_for_snapshot(self) -> None:", 1)[1]
    snapshot_block = snapshot_block.split("@modal.enter(snap=False)", 1)[0]
    assert "from comfyui_runtime_engine.modal import ModalSnapshotAdapter" in snapshot_block
    assert snapshot_block.index("from comfyui_runtime_engine.modal import ModalSnapshotAdapter") < snapshot_block.index("ModalSnapshotAdapter(")
