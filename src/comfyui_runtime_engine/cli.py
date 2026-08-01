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
    commands.add_parser("doctor", help="Inspect ComfyUI compatibility")
    commands.add_parser("simulate", help="Run embedded lifecycle simulation locally")
    commands.add_parser("probe", help="Bootstrap real ComfyUI in-process without binding the HTTP port")
    start = commands.add_parser("start", help="Start runtime")
    start.add_argument("--mode", choices=("normal", "embedded", "embedded-simulation"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = RuntimeConfig.from_toml(args.config)
        engine = RuntimeEngine(config)
        if args.command == "doctor":
            print(json.dumps(engine.doctor(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "simulate":
            engine.simulate()
            return 0
        if args.command == "probe":
            print(json.dumps(engine.probe_embedded(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "start":
            return engine.start(args.mode)
        return 2
    except (RuntimeEngineError, OSError, ValueError) as exc:
        print(f"comfy-runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
