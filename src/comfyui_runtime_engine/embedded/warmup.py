from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from ..errors import RuntimeEngineError
from ..warmup import WarmupWorkflow


class EmbeddedWarmupRunner:
    def __init__(self, bootstrap: object) -> None:
        self.bootstrap = bootstrap

    def run(
        self,
        workflow: WarmupWorkflow,
        *,
        timeout_seconds: float,
        hold_seconds: float,
    ) -> dict[str, Any]:
        handle = self.bootstrap.bootstrap()
        loop = handle.loop
        server_task = loop.create_task(handle.start_all())
        started = time.monotonic()

        try:
            self._wait_for_server(loop, server_task, timeout_seconds)
            # torch is imported only after ComfyUI completed its own bootstrap.
            import torch
            import comfy.model_management as model_management

            before = self._vram(torch)
            loaded_before = self._loaded_models(model_management)
            prompt_id = self._submit(workflow.prompt)
            history = self._wait_history(loop, prompt_id, timeout_seconds)
            torch.cuda.synchronize()
            after = self._vram(torch)
            loaded_after = self._loaded_models(model_management)

            if hold_seconds > 0:
                deadline = time.monotonic() + hold_seconds
                while time.monotonic() < deadline:
                    loop.run_until_complete(asyncio.sleep(0.05))

            report = {
                "pid": os.getpid(),
                "same_process": True,
                "workflow": str(workflow.path),
                "node_count": workflow.node_count,
                "prompt_id": prompt_id,
                "history_status": self._history_status(history),
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "vram_before": before,
                "vram_after": after,
                "vram_delta_allocated_bytes": (
                    after["memory_allocated_bytes"] - before["memory_allocated_bytes"]
                ),
                "vram_delta_reserved_bytes": (
                    after["memory_reserved_bytes"] - before["memory_reserved_bytes"]
                ),
                "loaded_models_before": loaded_before,
                "loaded_models_after": loaded_after,
                "loaded_model_count_delta": len(loaded_after) - len(loaded_before),
                "success": self._history_status(history) == "success",
            }
            self.bootstrap._events.emit("warmup.completed", **report)
            return report
        finally:
            server_task.cancel()
            if not server_task.done():
                loop.run_until_complete(
                    asyncio.gather(server_task, return_exceptions=True)
                )
            self.bootstrap.close()

    def _wait_for_server(self, loop, task, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            loop.run_until_complete(asyncio.sleep(0.05))
            if task.done():
                error = task.exception()
                if error:
                    raise RuntimeEngineError(f"Embedded server failed: {error}")
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
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise RuntimeEngineError(f"Warmup response has no prompt_id: {body}")
        return str(prompt_id)

    def _wait_history(self, loop, prompt_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        url = f"http://{self.bootstrap._host}:{self.bootstrap._port}/history/{prompt_id}"
        while time.monotonic() < deadline:
            loop.run_until_complete(asyncio.sleep(0.2))
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, json.JSONDecodeError):
                continue
            item = body.get(prompt_id)
            if isinstance(item, dict):
                status = item.get("status", {})
                if status.get("completed") is True:
                    return item
        raise RuntimeEngineError(
            f"Warmup workflow timed out after {timeout:.1f} seconds"
        )

    @staticmethod
    def _history_status(history: dict[str, Any]) -> str:
        status = history.get("status", {})
        messages = status.get("messages", [])
        for message in messages:
            if isinstance(message, list) and message and message[0] == "execution_error":
                return "error"
        return "success" if status.get("completed") else "unknown"

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
            list(getattr(model_management, "current_loaded_models", []))
        ):
            model = getattr(loaded, "model", None)
            result.append(
                {
                    "index": index,
                    "loaded_type": type(loaded).__name__,
                    "model_type": type(model).__name__ if model is not None else None,
                    "device": str(getattr(loaded, "device", "")),
                    "weights_loaded": bool(
                        getattr(loaded, "weights_loaded", False)
                    ),
                    "currently_used": bool(
                        getattr(loaded, "currently_used", False)
                    ),
                }
            )
        return result
