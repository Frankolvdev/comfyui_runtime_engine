from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import socket
import threading
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditFinding:
    component: str
    category: str
    disposition: str
    severity: str
    detected: bool
    details: dict[str, Any]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SnapshotCompatibilityAudit:
    """Inspect the live embedded runtime immediately before a provider snapshot.

    This audit is deliberately conservative. It does not claim that arbitrary
    Python objects are snapshot-safe. It verifies the state we explicitly rely
    on and marks operating-system resources for reconstruction after restore.
    """

    SAFE = "preserve"
    RECREATE = "recreate_after_restore"
    VERIFY = "preserve_then_verify"

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path.expanduser().resolve()

    def run(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        prompt_server: Any,
        prompt_queue: Any,
        residency_guard: dict[str, Any],
        snapshot_state: dict[str, Any] | None,
        model_paths: dict[str, Any],
    ) -> dict[str, Any]:
        findings: list[AuditFinding] = []

        findings.append(self._python_state())
        findings.append(self._event_loop(loop))
        findings.append(self._prompt_server(prompt_server))
        findings.append(self._prompt_queue(prompt_queue))
        findings.append(self._threads())
        findings.append(self._asyncio_tasks(loop))
        findings.append(self._socketpair())
        findings.append(self._cuda_state(residency_guard, snapshot_state))
        findings.append(self._model_paths(model_paths))
        findings.append(self._gc_state())

        blockers = [
            item.component
            for item in findings
            if item.severity == "blocker" and item.detected
        ]
        recreate = [
            item.component
            for item in findings
            if item.disposition == self.RECREATE and item.detected
        ]
        verify_after_restore = [
            item.component
            for item in findings
            if item.disposition == self.VERIFY and item.detected
        ]

        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "findings": [item.to_dict() for item in findings],
            "summary": {
                "finding_count": len(findings),
                "blockers": blockers,
                "recreate_after_restore": recreate,
                "verify_after_restore": verify_after_restore,
                "snapshot_compatible": not blockers,
            },
            "after_restore_plan": {
                "required_steps": [
                    "create a fresh asyncio event loop when the provider requires it",
                    "rebind aiohttp HTTP sockets instead of trusting captured listeners",
                    "restart PromptServer network serving on the restored loop",
                    "restart background workers and executors that are not alive",
                    "verify CUDA availability and resident model identities",
                    "verify model paths and input/output directories",
                    "run a lightweight health probe before accepting the first job",
                ],
                "must_not_repeat": [
                    "reload configured resident model weights from disk",
                    "rerun the full warmup workflow when all resident identities are restored",
                    "modify the user workflow",
                ],
            },
        }
        self.write(report)
        return report

    def write(self, report: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.output_path)

    @staticmethod
    def _python_state() -> AuditFinding:
        return AuditFinding(
            component="python_interpreter",
            category="process_memory",
            disposition=SnapshotCompatibilityAudit.SAFE,
            severity="info",
            detected=True,
            details={
                "pid": os.getpid(),
                "thread_name": threading.current_thread().name,
            },
            recommendation=(
                "Preserve imported modules and ordinary Python heap state."
            ),
        )

    @staticmethod
    def _event_loop(loop: asyncio.AbstractEventLoop) -> AuditFinding:
        closed = loop.is_closed()
        running = loop.is_running()
        return AuditFinding(
            component="asyncio_event_loop",
            category="os_runtime_resource",
            disposition=SnapshotCompatibilityAudit.RECREATE,
            severity="blocker" if closed else "warning",
            detected=True,
            details={
                "type": type(loop).__name__,
                "closed": closed,
                "running": running,
                "debug": loop.get_debug(),
            },
            recommendation=(
                "Do not rely on captured selector/proactor internals. Recreate or "
                "reattach the event loop in after_restore before serving requests."
            ),
        )

    @staticmethod
    def _prompt_server(prompt_server: Any) -> AuditFinding:
        sockets = []
        runner = getattr(prompt_server, "runner", None)
        site = getattr(prompt_server, "site", None)
        for candidate in (runner, site):
            server = getattr(candidate, "_server", None)
            for sock in getattr(server, "sockets", []) or []:
                try:
                    sockets.append(str(sock.getsockname()))
                except OSError:
                    sockets.append("<closed>")
        return AuditFinding(
            component="prompt_server_network",
            category="network_resource",
            disposition=SnapshotCompatibilityAudit.RECREATE,
            severity="warning",
            detected=True,
            details={
                "type": type(prompt_server).__name__,
                "runner_present": runner is not None,
                "site_present": site is not None,
                "bound_sockets": sockets,
            },
            recommendation=(
                "Preserve routes and Python server objects, but close/rebind HTTP and "
                "WebSocket listeners after restore."
            ),
        )

    @staticmethod
    def _prompt_queue(prompt_queue: Any) -> AuditFinding:
        return AuditFinding(
            component="prompt_queue",
            category="python_state",
            disposition=SnapshotCompatibilityAudit.VERIFY,
            severity="info",
            detected=True,
            details={
                "type": type(prompt_queue).__name__,
                "queue_size": len(getattr(prompt_queue, "queue", []) or []),
                "history_size": len(getattr(prompt_queue, "history", {}) or {}),
            },
            recommendation=(
                "Preserve queue structures only when empty at snapshot time; verify "
                "that no in-flight job was captured."
            ),
        )

    @staticmethod
    def _threads() -> AuditFinding:
        threads = [
            {
                "name": thread.name,
                "ident": thread.ident,
                "daemon": thread.daemon,
                "alive": thread.is_alive(),
            }
            for thread in threading.enumerate()
        ]
        non_main = [item for item in threads if item["name"] != "MainThread"]
        return AuditFinding(
            component="background_threads",
            category="os_runtime_resource",
            disposition=SnapshotCompatibilityAudit.RECREATE,
            severity="warning",
            detected=bool(non_main),
            details={
                "count": len(threads),
                "non_main_count": len(non_main),
                "threads": threads,
            },
            recommendation=(
                "After restore, verify every required worker thread and restart missing "
                "PromptExecutor/custom-node workers. Never assume thread liveness."
            ),
        )

    @staticmethod
    def _asyncio_tasks(loop: asyncio.AbstractEventLoop) -> AuditFinding:
        try:
            tasks = list(asyncio.all_tasks(loop))
        except RuntimeError:
            tasks = []
        task_details = []
        for task in tasks:
            coroutine = task.get_coro()
            task_details.append(
                {
                    "name": task.get_name(),
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                    "coroutine": getattr(coroutine, "__qualname__", type(coroutine).__name__),
                }
            )
        active = [item for item in task_details if not item["done"]]
        return AuditFinding(
            component="asyncio_tasks",
            category="scheduled_runtime_state",
            disposition=SnapshotCompatibilityAudit.RECREATE,
            severity="warning",
            detected=bool(active),
            details={
                "total": len(task_details),
                "active": len(active),
                "tasks": task_details,
            },
            recommendation=(
                "Cancel or quiesce nonessential tasks before snapshot. Recreate serving, "
                "manager refresh, timers and background tasks after restore."
            ),
        )

    @staticmethod
    def _socketpair() -> AuditFinding:
        left = right = None
        success = True
        error = None
        try:
            left, right = socket.socketpair()
        except OSError as exc:
            success = False
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if left is not None:
                left.close()
            if right is not None:
                right.close()
        return AuditFinding(
            component="socket_subsystem",
            category="os_runtime_resource",
            disposition=SnapshotCompatibilityAudit.VERIFY,
            severity="blocker" if not success else "info",
            detected=True,
            details={"socketpair_ok": success, "error": error},
            recommendation=(
                "Verify socket creation after restore before exposing the container."
            ),
        )

    @staticmethod
    def _cuda_state(
        residency_guard: dict[str, Any],
        snapshot_state: dict[str, Any] | None,
    ) -> AuditFinding:
        protected = list(residency_guard.get("protected_loaded_models", []))
        all_cuda = bool(protected) and all(
            str(item.get("device", "")).startswith("cuda") for item in protected
        )
        all_full = bool(protected) and all(
            int(item.get("loaded_bytes", 0))
            >= int(item.get("total_model_bytes", 0))
            > 0
            for item in protected
        )
        ready = bool(snapshot_state and snapshot_state.get("snapshot_ready"))
        success = all_cuda and all_full and ready
        return AuditFinding(
            component="cuda_resident_models",
            category="gpu_memory",
            disposition=SnapshotCompatibilityAudit.VERIFY,
            severity="blocker" if not success else "info",
            detected=True,
            details={
                "protected_count": len(protected),
                "all_cuda": all_cuda,
                "all_fully_loaded": all_full,
                "snapshot_state_ready": ready,
                "models": protected,
            },
            recommendation=(
                "Preserve CUDA tensors and ModelPatchers, then compare resident "
                "identities and loaded bytes in after_restore."
            ),
        )

    @staticmethod
    def _model_paths(model_paths: dict[str, Any]) -> AuditFinding:
        registered = list(model_paths.get("registered", []))
        missing = list(model_paths.get("missing_directories", []))
        return AuditFinding(
            component="comfyui_model_paths",
            category="python_configuration",
            disposition=SnapshotCompatibilityAudit.VERIFY,
            severity="warning" if not registered else "info",
            detected=True,
            details={
                "registered_count": len(registered),
                "missing_count": len(missing),
                "registered": registered,
                "missing": missing,
            },
            recommendation=(
                "Preserve folder_paths configuration and verify mounted paths exist "
                "after restore before accepting jobs."
            ),
        )

    @staticmethod
    def _gc_state() -> AuditFinding:
        gc.collect()
        return AuditFinding(
            component="python_gc",
            category="process_memory",
            disposition=SnapshotCompatibilityAudit.SAFE,
            severity="info",
            detected=True,
            details={
                "tracked_objects": len(gc.get_objects()),
                "garbage_count": len(gc.garbage),
            },
            recommendation=(
                "Run garbage collection immediately before the provider snapshot."
            ),
        )
