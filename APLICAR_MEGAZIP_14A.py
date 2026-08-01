from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "src" / "comfyui_runtime_engine" / "embedded" / "warmup.py"
CLI = ROOT / "src" / "comfyui_runtime_engine" / "cli.py"


def patch_runner() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    signature_old = (
        "        snapshot_lifecycle=None,\n"
        "        snapshot_audit: SnapshotCompatibilityAudit | None = None,\n"
        "    ) -> dict[str, Any]:"
    )
    signature_new = (
        "        snapshot_lifecycle=None,\n"
        "        snapshot_audit: SnapshotCompatibilityAudit | None = None,\n"
        "        persist_after_ready: bool = False,\n"
        "        persistent_ready_callback=None,\n"
        "    ) -> dict[str, Any]:"
    )
    if signature_new not in text:
        if signature_old not in text:
            raise RuntimeError("No se encontró la firma esperada de EmbeddedWarmupRunner.run")
        text = text.replace(signature_old, signature_new, 1)

    emit_old = (
        '            self.bootstrap._events.emit("warmup.completed", **report)\n'
        "            return report\n"
    )
    emit_new = (
        '            self.bootstrap._events.emit("warmup.completed", **report)\n'
        "            if persistent_ready_callback is not None:\n"
        "                persistent_ready_callback(report)\n"
        "            if persist_after_ready:\n"
        "                stop_requested = False\n"
        "\n"
        "                def request_stop(_signum, _frame):\n"
        "                    nonlocal stop_requested\n"
        "                    stop_requested = True\n"
        "\n"
        "                import signal\n"
        "                previous_sigterm = signal.signal(signal.SIGTERM, request_stop)\n"
        "                previous_sigint = signal.signal(signal.SIGINT, request_stop)\n"
        "                try:\n"
        "                    while not stop_requested:\n"
        "                        loop.run_until_complete(asyncio.sleep(0.25))\n"
        "                finally:\n"
        "                    signal.signal(signal.SIGTERM, previous_sigterm)\n"
        "                    signal.signal(signal.SIGINT, previous_sigint)\n"
        "            return report\n"
    )
    if emit_new not in text:
        if emit_old not in text:
            raise RuntimeError("No se encontró el bloque final esperado del warmup runner")
        text = text.replace(emit_old, emit_new, 1)

    RUNNER.write_text(text, encoding="utf-8")


def patch_cli() -> None:
    text = CLI.read_text(encoding="utf-8")
    if "from .modal.persistent import PersistentModalRuntime" not in text:
        text = text.replace(
            "from .runtime import RuntimeEngine\n",
            "from .runtime import RuntimeEngine\n"
            "from .modal.persistent import PersistentModalRuntime\n",
            1,
        )

    parser_anchor = '    snapshot_commands.add_parser("audit-inspect")\n'
    parser_insert = (
        parser_anchor
        + "\n"
        + '    modal_command = commands.add_parser("modal")\n'
        + "    modal_commands = modal_command.add_subparsers(\n"
        + '        dest="modal_command", required=True\n'
        + "    )\n"
        + '    modal_serve = modal_commands.add_parser("serve")\n'
        + '    modal_serve.add_argument("--ready-path", required=True)\n'
    )
    if 'commands.add_parser("modal")' not in text:
        if parser_anchor not in text:
            raise RuntimeError("No se encontró el bloque snapshot del CLI")
        text = text.replace(parser_anchor, parser_insert, 1)

    dispatch_anchor = (
        "        if args.command == \"models\":\n"
    )
    dispatch_insert = (
        "        if args.command == \"modal\":\n"
        "            return PersistentModalRuntime(\n"
        "                engine,\n"
        "                ready_path=Path(args.ready_path),\n"
        "            ).serve()\n"
        "        if args.command == \"models\":\n"
    )
    if 'if args.command == "modal":' not in text:
        if dispatch_anchor not in text:
            raise RuntimeError("No se encontró el dispatcher del CLI")
        text = text.replace(dispatch_anchor, dispatch_insert, 1)

    if "from pathlib import Path" not in text:
        text = text.replace(
            "import json\n",
            "import json\nfrom pathlib import Path\n",
            1,
        )

    CLI.write_text(text, encoding="utf-8")


def main() -> int:
    patch_runner()
    patch_cli()
    print("MegaZIP 14A aplicado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
