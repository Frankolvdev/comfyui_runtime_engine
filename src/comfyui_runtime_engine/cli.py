from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import RuntimeConfig
from .errors import RuntimeEngineError
from .runtime import RuntimeEngine
from .modal.persistent import PersistentModalRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfy-runtime")
    parser.add_argument("--config", default="runtime-engine.toml")
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models")
    model_commands = models.add_subparsers(
        dest="models_command", required=True
    )
    for name in ("scan", "plan", "verify"):
        model_commands.add_parser(name)

    warmup = commands.add_parser("warmup")
    warmup_commands = warmup.add_subparsers(
        dest="warmup_command", required=True
    )
    warmup_commands.add_parser("verify")
    probe = warmup_commands.add_parser("probe")
    probe.add_argument("--hold-seconds", type=float, default=10.0)
    probe.add_argument("--iterations", type=int, default=2)

    snapshot = commands.add_parser("snapshot")
    snapshot_commands = snapshot.add_subparsers(
        dest="snapshot_command", required=True
    )
    simulate = snapshot_commands.add_parser("simulate")
    simulate.add_argument("--hold-seconds", type=float, default=10.0)
    simulate.add_argument("--audit", action="store_true")
    snapshot_commands.add_parser("inspect")
    snapshot_commands.add_parser("audit-inspect")

    modal_command = commands.add_parser("modal")
    modal_commands = modal_command.add_subparsers(
        dest="modal_command", required=True
    )
    modal_serve = modal_commands.add_parser("serve")
    modal_serve.add_argument("--ready-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        engine = RuntimeEngine(RuntimeConfig.from_toml(args.config))
        if args.command == "modal":
            return PersistentModalRuntime(
                engine,
                ready_path=Path(args.ready_path),
            ).serve()
        if args.command == "models":
            report = {
                "scan": engine.residency_scan,
                "plan": engine.residency_plan,
                "verify": engine.residency_verify,
            }[args.models_command]()
        elif args.command == "warmup":
            if args.warmup_command == "verify":
                report = engine.warmup_verify()
            else:
                report = engine.warmup_probe(
                    hold_seconds=args.hold_seconds,
                    iterations=args.iterations,
                )
        elif args.snapshot_command == "simulate":
            report = engine.snapshot_simulate(
                hold_seconds=args.hold_seconds,
                audit=args.audit,
            )
        elif args.snapshot_command == "audit-inspect":
            report = engine.snapshot_audit_inspect()
        else:
            report = engine.snapshot_inspect()

        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if bool(report.get("success", True)) else 1
    except (RuntimeEngineError, OSError, ValueError, KeyError) as exc:
        print(f"comfy-runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
