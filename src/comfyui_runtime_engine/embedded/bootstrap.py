from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterator, Sequence

from ..adapters.base import ComfyInspection
from ..errors import CompatibilityError, RuntimeEngineError
from ..events import EventSink


@dataclass(slots=True)
class EmbeddedHandle:
    loop: asyncio.AbstractEventLoop
    prompt_server: object
    start_all: object
    module: ModuleType
    process_id: int

    @property
    def prompt_queue(self) -> object | None:
        return getattr(self.prompt_server, "prompt_queue", None)


class EmbeddedBootstrap:
    """Loads ComfyUI's public bootstrap entrypoint in the engine process.

    ComfyUI 0.15.x exposes ``start_comfyui(asyncio_loop)`` from ``main.py``.
    The engine calls that function rather than duplicating PromptServer, node
    initialization, the prompt worker, routes, or progress hooks.
    """

    def __init__(
        self,
        inspection: ComfyInspection,
        events: EventSink,
        *,
        host: str,
        port: int,
        extra_args: Sequence[str] = (),
    ) -> None:
        self._inspection = inspection
        self._events = events
        self._host = host
        self._port = port
        self._extra_args = tuple(extra_args)
        self._handle: EmbeddedHandle | None = None
        self._context: object | None = None

    def bootstrap(self) -> EmbeddedHandle:
        if self._handle is not None:
            raise RuntimeEngineError("Embedded ComfyUI is already bootstrapped in this process")

        self._events.emit(
            "embedded.import.start",
            root=str(self._inspection.root),
            main_file=str(self._inspection.main_file),
            pid=os.getpid(),
            args=list(self._effective_args()),
        )

        context = self._import_context()
        context.__enter__()
        self._context = context
        try:
            module = self._load_main_module()
            start_comfyui = getattr(module, "start_comfyui", None)
            if not callable(start_comfyui):
                raise CompatibilityError(
                    "ComfyUI main.py does not expose callable start_comfyui(asyncio_loop)"
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_result = start_comfyui(loop)
            if not isinstance(loop_result, tuple) or len(loop_result) != 3:
                raise CompatibilityError(
                    "ComfyUI start_comfyui() returned an unexpected value; expected "
                    "(event_loop, prompt_server, start_all)"
                )
            returned_loop, prompt_server, start_all = loop_result
            if returned_loop is not loop:
                raise CompatibilityError("ComfyUI replaced the engine-owned event loop")
            if not callable(start_all):
                raise CompatibilityError("ComfyUI start_all handle is not callable")
            if getattr(prompt_server, "prompt_queue", None) is None:
                raise CompatibilityError("PromptServer was created without prompt_queue")

            self._handle = EmbeddedHandle(
                loop=loop,
                prompt_server=prompt_server,
                start_all=start_all,
                module=module,
                process_id=os.getpid(),
            )
            self._events.emit(
                "embedded.bootstrap.ready",
                pid=os.getpid(),
                same_process=True,
                loop_type=type(loop).__name__,
                prompt_server_type=type(prompt_server).__name__,
                prompt_queue_type=type(prompt_server.prompt_queue).__name__,
                start_all_callable=True,
            )
            return self._handle
        except BaseException:
            self.close()
            raise

    def serve(self) -> int:
        handle = self.bootstrap()
        self._events.emit(
            "embedded.server.starting",
            host=self._host,
            port=self._port,
            pid=handle.process_id,
        )
        try:
            coroutine = handle.start_all()
            handle.loop.run_until_complete(coroutine)
            return 0
        except KeyboardInterrupt:
            self._events.emit("embedded.server.interrupted", pid=handle.process_id)
            return 0
        finally:
            self.close()

    def probe(self) -> dict[str, object]:
        handle = self.bootstrap()
        report = {
            "pid": handle.process_id,
            "same_process": handle.process_id == os.getpid(),
            "loop": type(handle.loop).__name__,
            "prompt_server": type(handle.prompt_server).__name__,
            "prompt_queue": type(handle.prompt_queue).__name__,
            "start_all_callable": callable(handle.start_all),
            "server_bound": False,
        }
        self._events.emit("embedded.probe.completed", **report)
        self.close()
        return report

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            cleanup_temp = getattr(handle.module, "cleanup_temp", None)
            if callable(cleanup_temp):
                try:
                    cleanup_temp()
                except Exception as exc:  # cleanup must not hide the main failure
                    self._events.emit("embedded.cleanup.warning", error=repr(exc))
            if not handle.loop.is_closed():
                pending = asyncio.all_tasks(handle.loop)
                for task in pending:
                    task.cancel()
                if pending:
                    handle.loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                handle.loop.close()
            self._events.emit("embedded.closed", pid=os.getpid())

        context = self._context
        self._context = None
        if context is not None:
            context.__exit__(None, None, None)

    def _effective_args(self) -> tuple[str, ...]:
        return (
            "--listen",
            self._host,
            "--port",
            str(self._port),
            "--disable-auto-launch",
            *self._extra_args,
        )

    @contextmanager
    def _import_context(self) -> Iterator[None]:
        root = self._inspection.root
        previous_cwd = Path.cwd()
        previous_path = list(sys.path)
        previous_argv = list(sys.argv)
        try:
            os.chdir(root)
            sys.path.insert(0, str(root))
            sys.argv = [str(self._inspection.main_file), *self._effective_args()]
            yield
        finally:
            sys.argv = previous_argv
            sys.path[:] = previous_path
            os.chdir(previous_cwd)

    def _load_main_module(self) -> ModuleType:
        module_name = "_comfyui_runtime_engine_embedded_main"
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, self._inspection.main_file)
        if spec is None or spec.loader is None:
            raise CompatibilityError(f"Cannot load ComfyUI main module: {self._inspection.main_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            raise CompatibilityError(
                f"Embedded bootstrap is missing dependency '{missing}'. Run the engine "
                "inside the same Python environment used by ComfyUI, or install the "
                "ComfyUI requirements in the engine environment."
            ) from exc
        return module
