from pathlib import Path

from comfyui_runtime_engine.embedded.warmup import EmbeddedWarmupRunner


class _Loaded:
    def __init__(self, model):
        self.model = model


class _Guard:
    def __init__(self, models):
        self._models = models

    def protected_loaded_models(self, model_management):
        return [_Loaded(model) for model in self._models]


class _ModelManagement:
    def __init__(self):
        self.calls = []

    def load_models_gpu(self, models, **kwargs):
        self.calls.append((list(models), dict(kwargs)))


def test_snapshot_residents_are_promoted_with_force_full_load():
    mm = _ModelManagement()
    resident_a = object()
    resident_b = object()
    EmbeddedWarmupRunner._force_residents_fully_loaded(
        mm,
        _Guard([resident_a, resident_b, resident_a]),
    )
    assert len(mm.calls) == 1
    models, kwargs = mm.calls[0]
    assert models == [resident_a, resident_b]
    assert kwargs == {"force_full_load": True}


def test_force_full_load_happens_before_snapshot_manifest_validation():
    source = Path("src/comfyui_runtime_engine/embedded/warmup.py").read_text(encoding="utf-8")
    assert source.index("self._force_residents_fully_loaded") < source.index("residency_guard = guard.report()")
    assert source.index("residency_guard = guard.report()") < source.index("snapshot_lifecycle.before_snapshot")
