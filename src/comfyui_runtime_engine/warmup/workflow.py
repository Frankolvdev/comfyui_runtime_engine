from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WarmupWorkflow:
    path: Path
    prompt: dict[str, Any]
    node_count: int

    @classmethod
    def load(cls, path: Path) -> "WarmupWorkflow":
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"Warmup workflow not found: {resolved}")
        raw = json.loads(resolved.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("Warmup workflow must be a JSON object")

        # Accept API-format directly or a wrapper containing {"prompt": {...}}.
        prompt = raw.get("prompt") if isinstance(raw.get("prompt"), dict) else raw
        if not prompt:
            raise ValueError("Warmup workflow contains no prompt nodes")

        for node_id, node in prompt.items():
            if not isinstance(node, dict) or "class_type" not in node:
                raise ValueError(
                    "Warmup workflow must be exported in ComfyUI API format; "
                    f"node {node_id!r} has no class_type"
                )
        return cls(resolved, prompt, len(prompt))
