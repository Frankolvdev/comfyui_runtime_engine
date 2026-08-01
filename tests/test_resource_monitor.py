from comfyui_runtime_engine.lifecycle import ResourceMonitor


def test_resource_monitor_can_create_socketpair() -> None:
    snapshot = ResourceMonitor.snapshot()
    assert snapshot.socketpair_ok is True
    assert snapshot.threads >= 1


def test_resource_compare_allows_small_thread_growth() -> None:
    before = ResourceMonitor.snapshot()
    after = ResourceMonitor.snapshot()
    report = ResourceMonitor.compare(before, after)
    assert report["clean"] is True
