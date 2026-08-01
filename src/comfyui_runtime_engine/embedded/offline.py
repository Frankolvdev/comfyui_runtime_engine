from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator


@contextmanager
def local_probe_environment(enabled: bool) -> Iterator[None]:
    """Reduce nonessential network/background work during short local probes.

    The variables are intentionally generic and reversible. Custom nodes that
    understand them may disable remote refreshes; nodes that do not will ignore
    them. The production server mode is not forced offline by this helper.
    """
    if not enabled:
        yield
        return

    overrides = {
        "COMFY_RUNTIME_PROBE": "1",
        "COMFY_RUNTIME_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
