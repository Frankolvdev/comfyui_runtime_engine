from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
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
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class EnvironmentReport:
    python_executable: str
    comfyui_root: str
    requirement_files: list[str]
    results: list[RequirementResult]
    pip_check: CheckResult | None
    import_checks: list[CheckResult]
    success: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "python_executable": self.python_executable,
            "comfyui_root": self.comfyui_root,
            "requirement_files": self.requirement_files,
            "results": [asdict(item) for item in self.results],
            "pip_check": asdict(self.pip_check) if self.pip_check else None,
            "import_checks": [asdict(item) for item in self.import_checks],
            "success": self.success,
        }


class EnvironmentManager:
    """Audits and synchronizes ComfyUI dependencies into the engine venv.

    Requirements are resolved in one pip invocation. This avoids the previous
    last-file-wins behavior where custom nodes repeatedly replaced NumPy,
    Transformers, OpenCV, and related packages.
    """

    _IMPORT_CHECKS = {
        "sqlalchemy": "sqlalchemy",
        "numpy": "numpy",
        "opencv": "cv2",
        "transformers": "transformers",
        "diffusers": "diffusers",
        "piexif": "piexif",
        "lark": "lark",
        "gguf": "gguf",
        "machineid": "machineid",
        "pywavelets": "pywt",
        "iopath": "iopath",
        "numba": "numba",
    }

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
            "workspace": self._workspace_audit(),
            "imports": [asdict(item) for item in self.verify_imports()],
        }
        self.events.emit("environment.audit.completed", **report)
        return report

    def sync(self, *, dry_run: bool = False, strict: bool = False) -> EnvironmentReport:
        requirement_files = self.discover()
        results: list[RequirementResult] = []
        command = [sys.executable, "-m", "pip", "install"]
        for requirement_file in requirement_files:
            command.extend(["-r", str(requirement_file)])

        self.events.emit(
            "environment.sync.started",
            python_executable=sys.executable,
            requirement_count=len(requirement_files),
            dry_run=dry_run,
            strategy="single-resolution",
        )

        if dry_run:
            results.append(
                RequirementResult(
                    path="<combined>",
                    return_code=0,
                    command=command,
                    status="dry-run",
                )
            )
            pip_check = None
            import_checks: list[CheckResult] = []
            success = True
        else:
            completed = subprocess.run(command, cwd=self.root, check=False)
            results.append(
                RequirementResult(
                    path="<combined>",
                    return_code=completed.returncode,
                    command=command,
                    status="installed" if completed.returncode == 0 else "failed",
                )
            )
            pip_check = self.pip_check()
            import_checks = self.verify_imports()
            success = completed.returncode == 0 and pip_check.ok
            if strict:
                success = success and all(item.ok for item in import_checks)

        report = EnvironmentReport(
            python_executable=sys.executable,
            comfyui_root=str(self.root),
            requirement_files=[str(path) for path in requirement_files],
            results=results,
            pip_check=pip_check,
            import_checks=import_checks,
            success=success,
        )
        self.events.emit(
            "environment.sync.completed",
            success=success,
            requirement_count=len(requirement_files),
            pip_check_ok=(pip_check.ok if pip_check else None),
            failed_imports=[item.name for item in import_checks if not item.ok],
        )
        return report

    def verify(self) -> dict[str, object]:
        pip_check = self.pip_check()
        imports = self.verify_imports()
        workspace = self._workspace_audit()
        report = {
            "pip_check": asdict(pip_check),
            "imports": [asdict(item) for item in imports],
            "workspace": workspace,
            "success": pip_check.ok and bool(workspace["writable"]),
        }
        self.events.emit("environment.verify.completed", **report)
        return report

    def pip_check(self) -> CheckResult:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )
        detail = (completed.stdout or completed.stderr or "").strip()
        return CheckResult("pip-check", completed.returncode == 0, detail or "No broken requirements found.")

    def verify_imports(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for name, module_name in self._IMPORT_CHECKS.items():
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, "__version__", None)
                results.append(CheckResult(name, True, str(version or "imported")))
            except Exception as exc:  # import failures are diagnostics, not engine crashes
                results.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))
        return results

    def ensure_workspace(self) -> dict[str, object]:
        created: list[str] = []
        for name in ("models", "input", "output", "temp", "user"):
            path = self.root / name
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(str(path))
        report = self._workspace_audit()
        report["created"] = created
        self.events.emit("environment.workspace.ready", **report)
        return report

    def _workspace_audit(self) -> dict[str, object]:
        checks: dict[str, bool] = {}
        for name in ("models", "input", "output", "temp", "user"):
            path = self.root / name
            checks[name] = path.is_dir() and self._can_write(path)
        return {
            "root": str(self.root),
            "paths": checks,
            "writable": all(checks.values()),
        }

    @staticmethod
    def _can_write(path: Path) -> bool:
        try:
            probe = path / ".comfy-runtime-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False
