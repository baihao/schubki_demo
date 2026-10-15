from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass(frozen=True)
class RigKeyframeRecord:
    index: int
    ts: int
    left_name: str
    right_name: str
    left_src: Path
    right_src: Path


def load_keyframes_json(path: Path) -> List[RigKeyframeRecord]:
    raw: List[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        RigKeyframeRecord(
            index=int(row["index"]),
            ts=int(row["ts"]),
            left_name=str(row["left_name"]),
            right_name=str(row["right_name"]),
            left_src=Path(row["left_src"]),
            right_src=Path(row["right_src"]),
        )
        for row in raw
    ]
