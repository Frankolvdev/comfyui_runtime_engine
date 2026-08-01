from __future__ import annotations

from dataclasses import dataclass
import platform
import subprocess
import sys

from ..events import EventSink


@dataclass(frozen=True, slots=True)
class CUDAInstallPlan:
    torch_version: str = "2.11.0"
    torchvision_version: str = "0.26.0"
    torchaudio_version: str = "2.11.0"
    channel: str = "cu128"

    @property
    def index_url(self) -> str:
        return f"https://download.pytorch.org/whl/{self.channel}"

    def uninstall_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchvision",
            "torchaudio",
        ]

    def install_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            f"torch=={self.torch_version}",
            f"torchvision=={self.torchvision_version}",
            f"torchaudio=={self.torchaudio_version}",
            "--index-url",
            self.index_url,
        ]


class CUDAEnvironmentSetup:
    """Controlled PyTorch CUDA preparation for the local engine venv."""

    def __init__(self, events: EventSink) -> None:
        self.events = events

    def prepare(
        self,
        *,
        dry_run: bool = False,
        channel: str = "cu128",
    ) -> dict[str, object]:
        plan = CUDAInstallPlan(channel=channel)
        commands = [plan.uninstall_command(), plan.install_command()]
        platform_report = {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        }
        self.events.emit(
            "gpu.cuda_setup.started",
            dry_run=dry_run,
            channel=channel,
            commands=commands,
            platform=platform_report,
        )

        if platform.system() not in {"Windows", "Linux"}:
            report = {
                "dry_run": dry_run,
                "channel": channel,
                "commands": commands,
                "return_codes": [],
                "platform": platform_report,
                "success": False,
                "error": "CUDA PyTorch setup is supported only on Windows or Linux.",
            }
            self.events.emit("gpu.cuda_setup.completed", **report)
            return report

        if channel not in {"cu128", "cu130"}:
            report = {
                "dry_run": dry_run,
                "channel": channel,
                "commands": commands,
                "return_codes": [],
                "platform": platform_report,
                "success": False,
                "error": "Unsupported CUDA channel. Allowed: cu128, cu130.",
            }
            self.events.emit("gpu.cuda_setup.completed", **report)
            return report

        return_codes: list[int] = []
        if dry_run:
            return_codes = [0, 0]
        else:
            for command in commands:
                completed = subprocess.run(command, check=False)
                return_codes.append(completed.returncode)

        report = {
            "dry_run": dry_run,
            "channel": channel,
            "commands": commands,
            "return_codes": return_codes,
            "platform": platform_report,
            "success": all(code == 0 for code in return_codes),
            "next_command": "comfy-runtime --config runtime-engine.toml gpu-probe",
        }
        self.events.emit("gpu.cuda_setup.completed", **report)
        return report
