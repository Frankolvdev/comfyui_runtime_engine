from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True, slots=True)
class WarmupInput:
    source: Path
    target: str

    def validate(self) -> None:
        if not self.source.is_file():
            raise ValueError(f"Warmup input source not found: {self.source}")
        if not self.target or Path(self.target).is_absolute():
            raise ValueError(
                f"Warmup input target must be a relative filename: {self.target!r}"
            )
        if ".." in Path(self.target).parts:
            raise ValueError(
                f"Warmup input target cannot escape the input folder: {self.target!r}"
            )


class WarmupInputStager:
    def __init__(self, input_directory: Path) -> None:
        self.input_directory = input_directory.resolve()

    def stage(self, inputs: tuple[WarmupInput, ...]) -> dict[str, Any]:
        self.input_directory.mkdir(parents=True, exist_ok=True)
        staged: list[dict[str, object]] = []

        for item in inputs:
            item.validate()
            destination = (self.input_directory / item.target).resolve()
            try:
                destination.relative_to(self.input_directory)
            except ValueError as exc:
                raise ValueError(
                    f"Warmup input target escapes input folder: {item.target}"
                ) from exc

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, destination)
            staged.append(
                {
                    "source": str(item.source),
                    "target": item.target.replace("\\", "/"),
                    "destination": str(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )

        return {
            "input_directory": str(self.input_directory),
            "count": len(staged),
            "staged": staged,
        }
