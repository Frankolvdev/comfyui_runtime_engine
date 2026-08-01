from __future__ import annotations

from dataclasses import dataclass, asdict
import asyncio
import gc
import os
import socket
import threading
import time
from typing import Any


@dataclass(slots=True)
class ResourceSnapshot:
    pid: int
    threads: int
    thread_names: list[str]
    asyncio_tasks: int
    socketpair_ok: bool
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    """Small process-resource monitor used by local probes."""

    @staticmethod
    def snapshot(loop: asyncio.AbstractEventLoop | None = None) -> ResourceSnapshot:
        thread_names = sorted(thread.name for thread in threading.enumerate())
        task_count = 0
        if loop is not None and not loop.is_closed():
            try:
                task_count = len(asyncio.all_tasks(loop))
            except RuntimeError:
                task_count = 0

        socketpair_ok = True
        left = right = None
        try:
            left, right = socket.socketpair()
        except OSError:
            socketpair_ok = False
        finally:
            if left is not None:
                left.close()
            if right is not None:
                right.close()

        return ResourceSnapshot(
            pid=os.getpid(),
            threads=len(thread_names),
            thread_names=thread_names,
            asyncio_tasks=task_count,
            socketpair_ok=socketpair_ok,
            timestamp=time.time(),
        )

    @staticmethod
    def cleanup_gc() -> None:
        gc.collect()

    @staticmethod
    def compare(
        before: ResourceSnapshot,
        after: ResourceSnapshot,
        *,
        allowed_thread_growth: int = 2,
    ) -> dict[str, Any]:
        new_threads = sorted(set(after.thread_names) - set(before.thread_names))
        removed_threads = sorted(set(before.thread_names) - set(after.thread_names))
        thread_growth = after.threads - before.threads
        return {
            "before": before.to_dict(),
            "after": after.to_dict(),
            "thread_growth": thread_growth,
            "new_threads": new_threads,
            "removed_threads": removed_threads,
            "socketpair_ok": after.socketpair_ok,
            "clean": (
                thread_growth <= allowed_thread_growth
                and after.socketpair_ok
            ),
        }
