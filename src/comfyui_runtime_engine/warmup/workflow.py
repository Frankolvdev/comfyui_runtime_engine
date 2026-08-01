from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PURGE_KEYWORDS = (
    "purge",
    "unload",
    "free",
    "emptycache",
    "empty_cache",
    "vram",
    "clearcache",
    "clear_cache",
    "cleanup",
    "clean_gpu",
)


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

    def required_input_images(self) -> list[str]:
        result: list[str] = []
        for node in self.prompt.values():
            if node.get("class_type") != "LoadImage":
                continue
            value = node.get("inputs", {}).get("image")
            if isinstance(value, str) and value not in result:
                result.append(value)
        return result

    def purge_nodes(self) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for node_id, node in self.prompt.items():
            class_type = str(node.get("class_type", ""))
            title = str(node.get("_meta", {}).get("title", ""))
            haystack = f"{class_type} {title}".casefold().replace(" ", "")
            matched = sorted(
                keyword for keyword in PURGE_KEYWORDS if keyword in haystack
            )
            if matched:
                matches.append(
                    {
                        "node_id": str(node_id),
                        "class_type": class_type,
                        "title": title or None,
                        "matched_keywords": matched,
                        "inputs": node.get("inputs", {}),
                    }
                )
        return matches
