from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LifecyclePhase(str, Enum):
    CREATED = "created"
    INSPECTED = "inspected"
    INITIALIZING = "initializing"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


_ALLOWED_TRANSITIONS: dict[LifecyclePhase, set[LifecyclePhase]] = {
    LifecyclePhase.CREATED: {LifecyclePhase.INSPECTED, LifecyclePhase.FAILED},
    LifecyclePhase.INSPECTED: {LifecyclePhase.INITIALIZING, LifecyclePhase.FAILED},
    LifecyclePhase.INITIALIZING: {LifecyclePhase.READY, LifecyclePhase.FAILED},
    LifecyclePhase.READY: {LifecyclePhase.STOPPING, LifecyclePhase.FAILED},
    LifecyclePhase.STOPPING: {LifecyclePhase.STOPPED, LifecyclePhase.FAILED},
    LifecyclePhase.STOPPED: set(),
    LifecyclePhase.FAILED: set(),
}


@dataclass(slots=True)
class LifecycleState:
    phase: LifecyclePhase = LifecyclePhase.CREATED

    def transition(self, target: LifecyclePhase) -> None:
        if target not in _ALLOWED_TRANSITIONS[self.phase]:
            raise RuntimeError(f"Invalid lifecycle transition: {self.phase.value} -> {target.value}")
        self.phase = target
