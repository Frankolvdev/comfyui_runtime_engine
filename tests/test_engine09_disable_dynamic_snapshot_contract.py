from pathlib import Path

PERSISTENT = Path('src/comfyui_runtime_engine/modal/persistent.py').read_text(encoding='utf-8')
WARMUP = Path('src/comfyui_runtime_engine/embedded/warmup.py').read_text(encoding='utf-8')


def test_modal_snapshot_embedded_comfy_disables_dynamic_vram():
    assert '"--disable-dynamic-vram"' in PERSISTENT
    assert 'gpu_args_list.append("--disable-dynamic-vram")' in PERSISTENT


def test_force_full_load_remains_strict_and_diagnostic():
    assert 'force_full_load=True' in WARMUP
    assert 'Snapshot residents were configured but no tagged ComfyUI LoadedModel' in WARMUP
    assert 'Configured snapshot resident models are still not fully loaded on CUDA' in WARMUP
    assert 'loaded.model_loaded_memory()' in WARMUP
    assert 'loaded.model_memory()' in WARMUP
