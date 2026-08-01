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

    commands.add_parser("doctor", help="Inspect ComfyUI compatibility and Python/CUDA state")
    commands.add_parser("simulate", help="Run embedded lifecycle simulation locally")
    commands.add_parser("probe", help="Bootstrap real ComfyUI in-process without binding HTTP")
    commands.add_parser("gpu-probe", help="Verify PyTorch CUDA with a minimal GPU allocation")

    server_probe = commands.add_parser(
        "server-probe",
        help="Start embedded HTTP locally in the configured mode, then stop",
    )
    server_probe.add_argument("--hold-seconds", type=float, default=1.0)

    gpu_server_probe = commands.add_parser(
        "gpu-server-probe",
        help="Start embedded ComfyUI on GPU without loading models, then stop",
    )
    gpu_server_probe.add_argument("--hold-seconds", type=float, default=2.0)

    gpu = commands.add_parser("gpu", help="Prepare the local PyTorch CUDA environment")
    gpu_commands = gpu.add_subparsers(dest="gpu_command", required=True)
    gpu_setup = gpu_commands.add_parser(
        "setup",
        help="Replace CPU-only PyTorch with a coordinated CUDA build",
    )
    gpu_setup.add_argument("--dry-run", action="store_true")
    gpu_setup.add_argument("--channel", choices=("cu128", "cu130"), default="cu128")

    env = commands.add_parser("env", help="Audit, synchronize or repair requirements")
    env_commands = env.add_subparsers(dest="env_command", required=True)
    env_commands.add_parser("audit", help="List discovered requirements")
    env_commands.add_parser("verify", help="Run effective pip/import/workspace checks")
    env_commands.add_parser("prepare", help="Create writable workspace directories")

    sync = env_commands.add_parser(
        "sync",
        help="Install discovered requirements into this Python environment",
    )
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--strict", action="store_true")

    repair = env_commands.add_parser(
        "repair",
        help="Repair Hugging Face and OpenCV-contrib compatibility",
    )
    repair.add_argument("--dry-run", action="store_true")

    start = commands.add_parser("start", help="Start runtime")
    start.add_argument(
        "--mode",
        choices=("normal", "embedded", "embedded-simulation"),
    )
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
        if args.command == "gpu-probe":
            report = engine.gpu_probe()
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if bool(report["success"]) else 1
        if args.command == "server-probe":
            print(
                json.dumps(
                    engine.probe_server(args.hold_seconds),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "gpu-server-probe":
            report = engine.probe_gpu_server(args.hold_seconds)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if bool(report.get("server_bound")) else 1
        if args.command == "gpu":
            report = engine.gpu_setup(
                dry_run=args.dry_run,
                channel=args.channel,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if bool(report["success"]) else 1
        if args.command == "env":
            if args.env_command == "audit":
                print(json.dumps(engine.environment_audit(), indent=2, ensure_ascii=False))
                return 0
            if args.env_command == "verify":
                report = engine.environment_verify()
                print(json.dumps(report, indent=2, ensure_ascii=False))
                return 0 if bool(report["success"]) else 1
            if args.env_command == "prepare":
                report = engine.environment_prepare()
                print(json.dumps(report, indent=2, ensure_ascii=False))
                return 0 if bool(report["writable"]) else 1
            if args.env_command == "repair":
                report = engine.environment_repair(dry_run=args.dry_run)
                print(json.dumps(report, indent=2, ensure_ascii=False))
                return 0 if bool(report["success"]) else 1
            report = engine.environment_sync(
                dry_run=args.dry_run,
                strict=args.strict,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if bool(report["success"]) else 1
        if args.command == "start":
            return engine.start(args.mode)
        return 2
    except (RuntimeEngineError, OSError, ValueError) as exc:
        print(f"comfy-runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
