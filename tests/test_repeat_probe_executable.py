import sys


def test_repeat_probe_must_use_active_interpreter() -> None:
    assert sys.executable
    assert "python" in sys.executable.lower()
