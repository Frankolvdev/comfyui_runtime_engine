from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ResidentState:
    source_path: str
    loaded_bytes: int
    total_model_bytes: int
    device: str
    fully_loaded: bool
    identity: str
    residency_mode: str = "cuda_loaded_model"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SnapshotStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    @staticmethod
    def model_identity(source_path: str, size_bytes: int) -> str:
        payload = f"{Path(source_path).resolve()}|{size_bytes}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def build_manifest(
        self,
        *,
        provider: str,
        phase: str,
        residents: list[dict[str, Any]],
        vram: dict[str, Any],
        workflow: str,
        prompt_id: str | None,
    ) -> dict[str, Any]:
        normalized: list[ResidentState] = []
        for item in residents:
            source = str(item.get("source_path") or "")
            loaded = int(item.get("loaded_bytes") or 0)
            total = int(item.get("total_model_bytes") or 0)
            mode = str(item.get("residency_mode") or "cuda_loaded_model")
            normalized.append(
                ResidentState(
                    source_path=source,
                    loaded_bytes=loaded,
                    total_model_bytes=total,
                    device=str(item.get("device") or ""),
                    fully_loaded=bool(total > 0 and loaded >= total),
                    identity=self.model_identity(source, total),
                    residency_mode=mode,
                )
            )
        return {
            "schema_version": 1,
            "provider": provider,
            "phase": phase,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workflow": workflow,
            "prompt_id": prompt_id,
            "resident_count": len(normalized),
            "residents": [item.to_dict() for item in normalized],
            "vram": vram,
            "snapshot_ready": bool(normalized) and all(
                item.fully_loaded
                and (
                    item.device.startswith("cuda")
                    if item.residency_mode == "cuda_loaded_model"
                    else item.residency_mode == "snapshot_object"
                )
                for item in normalized
            ),
        }

    def write(self, manifest: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise ValueError(f"Snapshot state not found: {self.path}")
        return json.loads(self.path.read_text(encoding="utf-8"))
