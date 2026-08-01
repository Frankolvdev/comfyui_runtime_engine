from __future__ import annotations

import argparse
import json
import sys

from .config import RuntimeConfig
from .errors import RuntimeEngineError
from .runtime import RuntimeEngine


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        engine = RuntimeEngine(RuntimeConfig.from_toml(args.config))
        if args.command == "models":
            report = {
                "scan": engine.residency_scan,
                "plan": engine.residency_plan,
                "verify": engine.residency_verify,
            }[args.models_command]()
        elif args.warmup_command == "verify":
            report = engine.warmup_verify()
        else:
            report = engine.warmup_probe(
                hold_seconds=args.hold_seconds,
                iterations=args.iterations,
            )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if bool(report.get("success", True)) else 1
    except (RuntimeEngineError, OSError, ValueError, KeyError) as exc:
        print(f"comfy-runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
