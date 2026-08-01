from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class ModalRuntimePaths:
    config_path: Path
    ready_path: Path
    restore_path: Path
    log_path: Path

    @classmethod
    def from_values(
        cls,
        *,
        config_path: str | Path,
        ready_path: str | Path = "/tmp/comfy-runtime-modal-ready.json",
        restore_path: str | Path = "/tmp/comfy-runtime-modal-restore.json",
        log_path: str | Path = "/tmp/comfy-runtime-modal.log",
    ) -> "ModalRuntimePaths":
        return cls(
            config_path=Path(config_path).expanduser().resolve(),
            ready_path=Path(ready_path).expanduser().absolute(),
            restore_path=Path(restore_path).expanduser().absolute(),
            log_path=Path(log_path).expanduser().absolute(),
        )


@dataclass(frozen=True, slots=True)
class ModalHealthReport:
    success: bool
    phase: str
    pid: int | None
    server_ready: bool
    snapshot_ready: bool
    resident_count: int
    expected_resident_count: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AtomicJsonFile:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def wait(
        self,
        *,
        timeout_seconds: float,
        predicate,
        poll_seconds: float = 0.25,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.read()
            except (OSError, json.JSONDecodeError):
                last = {}
            if predicate(last):
                return last
            time.sleep(poll_seconds)
        raise TimeoutError(
            f"Timed out waiting for Modal runtime state at {self.path}. "
            f"Last state: {last}"
        )
