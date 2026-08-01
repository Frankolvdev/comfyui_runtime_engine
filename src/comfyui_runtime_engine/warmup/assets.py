from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True, slots=True)
class AssetLinkResult:
    name: str
    source: str
    expected: str
    status: str
    link_type: str | None
    source_exists: bool
    expected_exists: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AssetLinkManager:
    """Prepare runtime-visible model assets without editing the workflow."""

    def __init__(self, *, replace_existing: bool = False) -> None:
        self.replace_existing = replace_existing

    def ensure_file_link(
        self,
        *,
        name: str,
        source: Path,
        expected: Path,
        mode: str = "auto",
    ) -> AssetLinkResult:
        source = source.expanduser().resolve()
        expected = expected.expanduser().absolute()

        if not source.is_file():
            return AssetLinkResult(
                name=name,
                source=str(source),
                expected=str(expected),
                status="source_missing",
                link_type=None,
                source_exists=False,
                expected_exists=expected.exists(),
            )

        if source == expected.resolve() if expected.exists() else False:
            return AssetLinkResult(
                name=name,
                source=str(source),
                expected=str(expected),
                status="already_same_file",
                link_type="same-file",
                source_exists=True,
                expected_exists=True,
            )

        if expected.is_symlink():
            try:
                if expected.resolve() == source:
                    return AssetLinkResult(
                        name=name,
                        source=str(source),
                        expected=str(expected),
                        status="already_linked",
                        link_type="symlink",
                        source_exists=True,
                        expected_exists=True,
                    )
            except OSError:
                pass

        if expected.exists() or expected.is_symlink():
            if not self.replace_existing:
                return AssetLinkResult(
                    name=name,
                    source=str(source),
                    expected=str(expected),
                    status="existing_target_not_replaced",
                    link_type="regular-file" if expected.is_file() else "other",
                    source_exists=True,
                    expected_exists=True,
                )
            if expected.is_dir() and not expected.is_symlink():
                raise ValueError(f"Expected SAM3 target is a directory: {expected}")
            expected.unlink(missing_ok=True)

        expected.parent.mkdir(parents=True, exist_ok=True)

        attempts = [mode] if mode != "auto" else ["symlink", "hardlink"]
        errors: list[str] = []
        for link_type in attempts:
            try:
                if link_type == "symlink":
                    os.symlink(source, expected, target_is_directory=False)
                elif link_type == "hardlink":
                    os.link(source, expected)
                elif link_type == "copy":
                    shutil.copy2(source, expected)
                else:
                    raise ValueError(f"Unsupported asset link mode: {link_type}")
                return AssetLinkResult(
                    name=name,
                    source=str(source),
                    expected=str(expected),
                    status="created",
                    link_type=link_type,
                    source_exists=True,
                    expected_exists=True,
                )
            except OSError as exc:
                errors.append(f"{link_type}: {type(exc).__name__}: {exc}")

        raise OSError(
            f"Could not prepare {name} at {expected}. Attempts: " + " | ".join(errors)
        )
