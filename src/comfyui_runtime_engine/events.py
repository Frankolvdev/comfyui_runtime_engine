from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    name: str
    timestamp: str
    details: dict[str, Any]


class EventSink:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = Lock()

    def emit(self, name: str, **details: Any) -> RuntimeEvent:
        event = RuntimeEvent(
            name=name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
        )
        payload = json.dumps(asdict(event), ensure_ascii=False, sort_keys=True)
        print(f"[comfy-runtime] {payload}", flush=True)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self._path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        return event
