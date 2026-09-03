from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from ..errors import RuntimeEngineError
from ..snapshot import SnapshotCompatibilityAudit
from ..warmup import (
    ComfyModelPathRegistrar,
    SelectiveResidencyGuard,
    WarmupInputStager,
    WarmupWorkflow,
)


class EmbeddedWarmupRunner:
    def __init__(
        self,
        bootstrap: object,
        *,
        model_roots: tuple[Path, ...],
        warmup_inputs: tuple,
        resident_paths: tuple[Path, ...],
    ) -> None:
        self.bootstrap = bootstrap
        self.model_roots = model_roots
        self.warmup_inputs = warmup_inputs
        self.resident_paths = resident_paths

    def run(
        self,
        workflow: WarmupWorkflow,
        *,
        timeout_seconds: float,
        hold_seconds: float,
        iterations: int,
        asset_links: dict[str, Any],
        snapshot_lifecycle=None,
        snapshot_audit: SnapshotCompatibilityAudit | None = None,
        persist_after_ready: bool = False,
        persistent_ready_callback=None,
    ) -> dict[str, Any]:
        handle = self.bootstrap.bootstrap()
        loop = handle.loop
        model_paths = ComfyModelPathRegistrar(self.model_roots).register()

        import folder_paths

        input_report = WarmupInputStager(
            Path(folder_paths.get_input_directory())
        ).stage(self.warmup_inputs)
        self._validate_required_images(
            workflow.required_input_images(),
            input_report,
            Path(folder_paths.get_input_directory()),
        )

        server_task = loop.create_task(handle.start_all())
        started = time.monotonic()
        runs: list[dict[str, Any]] = []

        guard = SelectiveResidencyGuard(self.resident_paths)
        try:
            self._wait_for_server(loop, server_task, timeout_seconds)
            guard.install()

            import torch
            import comfy.model_management as model_management

            for index in range(iterations):
                before = self._vram(torch)
                loaded_before = self._loaded_models(model_management)
                run_started = time.monotonic()

                prompt_id = self._run_blocking(
                    loop, self._submit, workflow.prompt, timeout=45.0
                )
                history = self._wait_history(loop, prompt_id, timeout_seconds)
                torch.cuda.synchronize()

                after = self._vram(torch)
                loaded_after = self._loaded_models(model_management)
                runs.append(
                    {
                        "iteration": index + 1,
                        "prompt_id": prompt_id,
                        "history_status": self._history_status(history),
                        "history_errors": self._history_errors(history),
                        "elapsed_ms": int(
                            (time.monotonic() - run_started) * 1000
                        ),
                        "vram_before": before,
                        "vram_after": after,
                        "vram_delta_allocated_bytes": (
                            after["memory_allocated_bytes"]
                            - before["memory_allocated_bytes"]
                        ),
                        "vram_delta_reserved_bytes": (
                            after["memory_reserved_bytes"]
                            - before["memory_reserved_bytes"]
                        ),
                        "loaded_models_before": loaded_before,
                        "loaded_models_after": loaded_after,
                    }
                )

            if hold_seconds > 0:
                deadline = time.monotonic() + hold_seconds
                while time.monotonic() < deadline:
                    loop.run_until_complete(asyncio.sleep(0.05))

            # Snapshot residents must be physically resident on CUDA, not merely
            # staged by ComfyUI's dynamic-VRAM loader. The warmup workflow can
            # legitimately leave large Flux/CLIP patchers partially loaded even
            # though they were exercised. Before validating snapshot readiness,
            # explicitly promote ONLY the configured/tagged residents to a full
            # CUDA load. This preserves normal ComfyUI memory management for every
            # non-resident model.
            self._force_residents_fully_loaded(model_management, guard)
            torch.cuda.synchronize()

            residency_guard = guard.report()
            snapshot_state = None
            if snapshot_lifecycle is not None:
                last_run = runs[-1] if runs else {}
                snapshot_state = snapshot_lifecycle.before_snapshot(
                    residency_guard=residency_guard,
                    vram=last_run.get("vram_after", {}),
                    workflow=str(workflow.path),
                    prompt_id=last_run.get("prompt_id"),
                )

            snapshot_audit_report = None
            if snapshot_audit is not None:
                snapshot_audit_report = snapshot_audit.run(
                    loop=loop,
                    prompt_server=handle.prompt_server,
                    prompt_queue=handle.prompt_queue,
                    residency_guard=residency_guard,
                    snapshot_state=snapshot_state,
                    model_paths=model_paths,
                )

            success = (
                all(run["history_status"] == "success" for run in runs)
                and residency_guard["protected_loaded_count"]
                >= len(self.resident_paths)
            )
            if snapshot_audit_report is not None:
                success = (
                    success
                    and snapshot_audit_report["summary"]["snapshot_compatible"]
                )

            report = {
                "pid": os.getpid(),
                "same_process": True,
                "workflow": str(workflow.path),
                "node_count": workflow.node_count,
                "iterations": iterations,
                "runs": runs,
                "first_elapsed_ms": runs[0]["elapsed_ms"] if runs else None,
                "second_elapsed_ms": (
                    runs[1]["elapsed_ms"] if len(runs) > 1 else None
                ),
                "speedup_ratio": (
                    runs[0]["elapsed_ms"] / runs[1]["elapsed_ms"]
                    if len(runs) > 1 and runs[1]["elapsed_ms"] > 0
                    else None
                ),
                "model_paths": model_paths,
                "warmup_inputs": input_report,
                "required_input_images": workflow.required_input_images(),
                "purge_nodes": workflow.purge_nodes(),
                "workflow_modified": False,
                "asset_links": asset_links,
                "residency_guard": residency_guard,
                "snapshot_state": snapshot_state,
                "snapshot_audit": snapshot_audit_report,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "success": success,
            }
            self.bootstrap._events.emit("warmup.completed", **report)
            if persistent_ready_callback is not None:
                persistent_ready_callback(report)
            if persist_after_ready:
                stop_requested = False

                def request_stop(_signum, _frame):
                    nonlocal stop_requested
                    stop_requested = True

                import signal
                previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
                previous_sigint = signal.signal(signal.SIGINT, request_stop)
                try:
                    while not stop_requested:
                        loop.run_until_complete(asyncio.sleep(0.25))
                finally:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                    signal.signal(signal.SIGINT, previous_sigint)
            return report
        finally:
            guard.restore()
            server_task.cancel()
            if not server_task.done():
                loop.run_until_complete(
                    asyncio.gather(server_task, return_exceptions=True)
                )
            self.bootstrap.close()

    @staticmethod
    def _force_residents_fully_loaded(model_management, guard) -> None:
        protected = list(guard.protected_loaded_models(model_management))
        patchers = []
        for loaded in protected:
            patcher = getattr(loaded, "model", None)
            if patcher is not None and patcher not in patchers:
                patchers.append(patcher)
        if not patchers:
            return
        model_management.load_models_gpu(
            patchers,
            force_full_load=True,
        )

    def _wait_for_server(self, loop, task, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            loop.run_until_complete(asyncio.sleep(0.05))
            if task.done() and task.exception():
                raise RuntimeEngineError(
                    f"Embedded server failed: {task.exception()}"
                )
            if self.bootstrap._port_is_open():
                return
        raise RuntimeEngineError("Embedded warmup server did not become ready")

    def _submit(self, prompt: dict[str, Any]) -> str:
        payload = json.dumps(
            {"prompt": prompt, "client_id": f"runtime-warmup-{uuid.uuid4()}"}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"http://{self.bootstrap._host}:{self.bootstrap._port}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeEngineError(
                f"Warmup prompt rejected with HTTP {exc.code}: {detail}"
            ) from exc
        if body.get("error") or not body.get("prompt_id"):
            raise RuntimeEngineError(
                "Warmup prompt failed validation: "
                + json.dumps(body, ensure_ascii=False)
            )
        return str(body["prompt_id"])

    def _wait_history(
        self,
        loop,
        prompt_id: str,
        timeout: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            loop.run_until_complete(asyncio.sleep(0.2))
            body = self._run_blocking(
                loop, self._fetch_history, prompt_id, timeout=20.0
            )
            item = body.get(prompt_id)
            if isinstance(item, dict) and item.get("status", {}).get("completed"):
                return item
        raise RuntimeEngineError(
            f"Warmup workflow timed out after {timeout:.1f} seconds"
        )

    def _fetch_history(self, prompt_id: str) -> dict[str, Any]:
        url = (
            f"http://{self.bootstrap._host}:{self.bootstrap._port}"
            f"/history/{prompt_id}"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _run_blocking(loop, function, *args, timeout: float):
        future = loop.run_in_executor(None, function, *args)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                raise RuntimeEngineError(
                    f"Blocking warmup operation timed out after {timeout:.1f}s"
                )
            loop.run_until_complete(asyncio.sleep(0.05))
        try:
            return future.result()
        except RuntimeEngineError:
            raise
        except Exception as exc:
            raise RuntimeEngineError(
                f"Warmup operation failed: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _validate_required_images(
        required: list[str],
        input_report: dict[str, Any],
        input_directory: Path,
    ) -> None:
        staged = {item["target"] for item in input_report["staged"]}
        missing = [
            name
            for name in required
            if name not in staged and not (input_directory / name).is_file()
        ]
        if missing:
            raise RuntimeEngineError(
                "Warmup workflow requires missing input images: "
                + ", ".join(missing)
            )

    @staticmethod
    def _history_status(history: dict[str, Any]) -> str:
        for message in history.get("status", {}).get("messages", []):
            if isinstance(message, list) and message and message[0] == "execution_error":
                return "error"
        return "success" if history.get("status", {}).get("completed") else "unknown"

    @staticmethod
    def _history_errors(history: dict[str, Any]) -> list[Any]:
        return [
            message
            for message in history.get("status", {}).get("messages", [])
            if isinstance(message, list)
            and message
            and message[0] == "execution_error"
        ]

    @staticmethod
    def _vram(torch) -> dict[str, Any]:
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
        }

    @staticmethod
    def _loaded_models(model_management) -> list[dict[str, Any]]:
        result = []
        for index, loaded in enumerate(
            list(model_management.current_loaded_models)
        ):
            patcher = loaded.model
            result.append(
                {
                    "index": index,
                    "loaded_type": type(loaded).__name__,
                    "model_type": type(patcher).__name__,
                    "device": str(getattr(loaded, "device", "")),
                    "currently_used": bool(
                        getattr(loaded, "currently_used", False)
                    ),
                    "source_path": getattr(
                        patcher, "_comfy_runtime_source_path", None
                    ),
                    "resident": bool(
                        getattr(patcher, "_comfy_runtime_resident", False)
                    ),
                    "loaded_bytes": int(loaded.model_loaded_memory()),
                    "total_model_bytes": int(loaded.model_memory()),
                }
            )
        return result
