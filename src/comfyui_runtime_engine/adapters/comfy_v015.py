from __future__ import annotations

import ast
from pathlib import Path
import re

from .base import ComfyAdapter, ComfyInspection

_VERSION_RE = re.compile(r"(?:__version__|version)\s*=\s*['\"]([^'\"]+)['\"]", re.I)


class ComfyV015Adapter(ComfyAdapter):
    family = "0.15"

    def inspect(self, root: Path) -> ComfyInspection:
        required = (
            root / "main.py",
            root / "server.py",
            root / "execution.py",
            root / "nodes.py",
            root / "folder_paths.py",
            root / "comfy",
            root / "comfy_execution",
        )
        missing = tuple(path for path in required if not path.exists())
        return ComfyInspection(
            root=root,
            version=self._detect_version(root),
            main_file=root / "main.py",
            required_paths=required,
            missing_paths=missing,
        )

    def normal_command(
        self,
        *,
        root: Path,
        python_executable: str,
        host: str,
        port: int,
        extra_args: tuple[str, ...],
    ) -> list[str]:
        return [
            python_executable,
            str(root / "main.py"),
            "--listen",
            host,
            "--port",
            str(port),
            *extra_args,
        ]

    @staticmethod
    def _detect_version(root: Path) -> str:
        version_file = root / "comfyui_version.py"
        if version_file.is_file():
            text = version_file.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in {"__version__", "version"}:
                                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                    return node.value.value
            except SyntaxError:
                pass
            match = _VERSION_RE.search(text)
            if match:
                return match.group(1)
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", text)
            if match:
                return match.group(1)
        return "unknown"
