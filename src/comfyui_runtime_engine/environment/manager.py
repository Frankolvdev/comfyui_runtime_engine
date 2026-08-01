from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from ..events import EventSink


@dataclass(slots=True)
class RequirementResult:
    path: str
    return_code: int
    command: list[str]
    status: str


@dataclass(slots=True)
class EnvironmentReport:
    python_executable: str
    comfyui_root: str
    requirement_files: list[str]
    results: list[RequirementResult]
    success: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "python_executable": self.python_executable,
            "comfyui_root": self.comfyui_root,
            "requirement_files": self.requirement_files,
            "results": [asdict(item) for item in self.results],
            "success": self.success,
        }


class EnvironmentManager:
    """Audits and synchronizes ComfyUI dependencies into the engine venv.

    Only requirements files are executed. Arbitrary install scripts from custom
    nodes are intentionally not run by this first implementation.
    """

    _NAMES = ("requirements.txt", "requirements-cuda.txt", "requirements_windows.txt")

    def __init__(self, root: Path, events: EventSink) -> None:
        self.root = root.resolve()
        self.events = events

    def discover(self) -> list[Path]:
        candidates: list[Path] = []
        core = self.root / "requirements.txt"
        if core.is_file():
            candidates.append(core)

        custom_root = self.root / "custom_nodes"
        if custom_root.is_dir():
            discovered: set[Path] = set()
            for node_dir in sorted((p for p in custom_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
                for candidate in node_dir.rglob("requirements*.txt"):
                    try:
                        relative_depth = len(candidate.relative_to(node_dir).parts)
                    except ValueError:
                        continue
                    if relative_depth <= 4 and candidate.is_file():
                        discovered.add(candidate)
            candidates.extend(sorted(discovered, key=lambda p: str(p).lower()))
        return candidates

    def audit(self) -> dict[str, object]:
        requirement_files = self.discover()
        report = {
            "python_executable": sys.executable,
            "comfyui_root": str(self.root),
            "requirement_files": [str(path) for path in requirement_files],
            "count": len(requirement_files),
        }
        self.events.emit("environment.audit.completed", **report)
        return report

    def sync(self, *, dry_run: bool = False, strict: bool = False) -> EnvironmentReport:
        requirement_files = self.discover()
        results: list[RequirementResult] = []
        self.events.emit(
            "environment.sync.started",
            python_executable=sys.executable,
            requirement_count=len(requirement_files),
            dry_run=dry_run,
            strict=strict,
        )

        for requirement_file in requirement_files:
            command = [sys.executable, "-m", "pip", "install", "-r", str(requirement_file)]
            if dry_run:
                result = RequirementResult(
                    path=str(requirement_file),
                    return_code=0,
                    command=command,
                    status="dry-run",
                )
            else:
                completed = subprocess.run(command, cwd=self.root, check=False)
                status = "installed" if completed.returncode == 0 else "failed"
                result = RequirementResult(
                    path=str(requirement_file),
                    return_code=completed.returncode,
                    command=command,
                    status=status,
                )
            results.append(result)
            self.events.emit(
                "environment.requirement.completed",
                path=result.path,
                return_code=result.return_code,
                status=result.status,
            )
            if strict and result.return_code != 0:
                break

        success = all(item.return_code == 0 for item in results)
        report = EnvironmentReport(
            python_executable=sys.executable,
            comfyui_root=str(self.root),
            requirement_files=[str(path) for path in requirement_files],
            results=results,
            success=success,
        )
        self.events.emit("environment.sync.completed", success=success, processed=len(results))
        return report
