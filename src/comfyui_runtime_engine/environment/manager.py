from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import subprocess
import sys

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
    """Audits, synchronizes and repairs the shared ComfyUI engine environment."""

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

    # Deterministic compatibility set for the current ComfyUI 0.15.1 laboratory.
    _REPAIR_INSTALL = (
        "huggingface_hub>=0.34.0,<1.0",
        "transformers>=4.57.0,<5",
        "diffusers>=0.35.1,<0.36",
        "opencv-contrib-python==4.11.0.86",
    )
    _OPENCV_VARIANTS = (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    )
    _VIRTUAL_OPENCV_REQUIREMENTS = (
        "requires opencv-python, which is not installed.",
        "requires opencv-python-headless, which is not installed.",
    )

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
            for node_dir in sorted(
                (p for p in custom_root.iterdir() if p.is_dir()),
                key=lambda p: p.name.lower(),
            ):
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
            results.append(RequirementResult("<combined>", 0, command, "dry-run"))
            pip_check = None
            import_checks: list[CheckResult] = []
            success = True
        else:
            completed = subprocess.run(command, cwd=self.root, check=False)
            results.append(
                RequirementResult(
                    "<combined>",
                    completed.returncode,
                    command,
                    "installed" if completed.returncode == 0 else "failed",
                )
            )
            pip_check = self.pip_check()
            import_checks = self.verify_imports()
            success = completed.returncode == 0 and pip_check.ok
            if strict:
                success = success and all(item.ok for item in import_checks)

        report = EnvironmentReport(
            sys.executable,
            str(self.root),
            [str(path) for path in requirement_files],
            results,
            pip_check,
            import_checks,
            success,
        )
        self.events.emit(
            "environment.sync.completed",
            success=success,
            requirement_count=len(requirement_files),
            pip_check_ok=(pip_check.ok if pip_check else None),
            failed_imports=[item.name for item in import_checks if not item.ok],
        )
        return report

    def repair(self, *, dry_run: bool = False) -> dict[str, object]:
        """Repair HF ecosystem and OpenCV-contrib compatibility deterministically."""
        uninstall = [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            *self._OPENCV_VARIANTS,
            "huggingface-hub",
            "transformers",
            "diffusers",
        ]
        install = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            *self._REPAIR_INSTALL,
        ]
        commands = [uninstall, install]
        self.events.emit("environment.repair.started", dry_run=dry_run, commands=commands)

        return_codes: list[int] = []
        if dry_run:
            return_codes = [0, 0]
        else:
            for command in commands:
                completed = subprocess.run(command, cwd=self.root, check=False)
                return_codes.append(completed.returncode)
            importlib.invalidate_caches()

        pip_check = self.pip_check() if not dry_run else None
        imports = self.verify_imports() if not dry_run else []
        opencv_features = self.verify_opencv_features() if not dry_run else []
        critical = {item.name: item.ok for item in imports}
        success = (
            all(code == 0 for code in return_codes)
            and (pip_check.ok if pip_check else True)
            and critical.get("transformers", True)
            and critical.get("diffusers", True)
            and all(item.ok for item in opencv_features)
        )
        report = {
            "dry_run": dry_run,
            "commands": commands,
            "return_codes": return_codes,
            "pip_check": asdict(pip_check) if pip_check else None,
            "imports": [asdict(item) for item in imports],
            "opencv_features": [asdict(item) for item in opencv_features],
            "success": success,
        }
        self.events.emit("environment.repair.completed", **report)
        return report

    def verify(self) -> dict[str, object]:
        # temp can be removed by ComfyUI after a probe, so recreate workspace first.
        workspace = self.ensure_workspace()
        pip_check = self.pip_check()
        imports = self.verify_imports()
        opencv_features = self.verify_opencv_features()

        critical_names = {"transformers", "diffusers"}
        critical_imports_ok = all(
            item.ok for item in imports if item.name in critical_names
        )
        report = {
            "pip_check": asdict(pip_check),
            "imports": [asdict(item) for item in imports],
            "opencv_features": [asdict(item) for item in opencv_features],
            "workspace": workspace,
            "success": (
                pip_check.ok
                and bool(workspace["writable"])
                and critical_imports_ok
                and all(item.ok for item in opencv_features)
            ),
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
        raw_detail = (completed.stdout or completed.stderr or "").strip()
        if completed.returncode == 0:
            return CheckResult("pip-check", True, raw_detail or "No broken requirements found.")

        lines = [line.strip() for line in raw_detail.splitlines() if line.strip()]
        effective_errors = [
            line for line in lines
            if not any(marker in line for marker in self._VIRTUAL_OPENCV_REQUIREMENTS)
        ]
        opencv_ok = all(item.ok for item in self.verify_opencv_features())
        if lines and not effective_errors and opencv_ok:
            return CheckResult(
                "pip-check",
                True,
                "Only virtual OpenCV distribution-name conflicts remain; "
                "opencv-contrib-python provides cv2 and guidedFilter. Original output:\n"
                + raw_detail,
            )

        return CheckResult(
            "pip-check",
            False,
            "\n".join(effective_errors) if effective_errors else raw_detail,
        )

    def verify_imports(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for name, module_name in self._IMPORT_CHECKS.items():
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, "__version__", None)
                results.append(CheckResult(name, True, str(version or "imported")))
            except Exception as exc:
                results.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))
        return results

    def verify_opencv_features(self) -> list[CheckResult]:
        try:
            cv2 = importlib.import_module("cv2")
            ximgproc = getattr(cv2, "ximgproc", None)
            guided_filter = getattr(ximgproc, "guidedFilter", None) if ximgproc is not None else None
            return [
                CheckResult(
                    "opencv.ximgproc.guidedFilter",
                    callable(guided_filter),
                    "available" if callable(guided_filter)
                    else "missing; install opencv-contrib-python",
                )
            ]
        except Exception as exc:
            return [
                CheckResult(
                    "opencv.ximgproc.guidedFilter",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            ]

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
