from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ComfyInspection:
    root: Path
    version: str
    main_file: Path
    required_paths: tuple[Path, ...]
    missing_paths: tuple[Path, ...]

    @property
    def structurally_valid(self) -> bool:
        return not self.missing_paths


class ComfyAdapter(ABC):
    family: str

    @abstractmethod
    def inspect(self, root: Path) -> ComfyInspection:
        raise NotImplementedError

    @abstractmethod
    def normal_command(
        self,
        *,
        root: Path,
        python_executable: str,
        host: str,
        port: int,
        extra_args: tuple[str, ...],
    ) -> list[str]:
        raise NotImplementedError
