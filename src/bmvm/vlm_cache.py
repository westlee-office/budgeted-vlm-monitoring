from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class VLMRecord:
    episode_id: str
    stream_id: str
    t_s: float
    score: float
    summary: str = ""
    model: str = ""
    prompt_hash: str = ""


class VLMCache:
    """JSONL cache for expensive VLM verifier outputs."""

    def __init__(self, records: Iterable[VLMRecord]) -> None:
        self.records: Dict[Tuple[str, str, float], VLMRecord] = {
            (r.episode_id, r.stream_id, round(r.t_s, 6)): r for r in records
        }

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "VLMCache":
        records = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append(
                    VLMRecord(
                        episode_id=row["episode_id"],
                        stream_id=row["stream_id"],
                        t_s=float(row["t_s"]),
                        score=float(row["score"]),
                        summary=row.get("summary", ""),
                        model=row.get("model", ""),
                        prompt_hash=row.get("prompt_hash", ""),
                    )
                )
        return cls(records)

    def get(self, episode_id: str, stream_id: str, t_s: float) -> Optional[VLMRecord]:
        return self.records.get((episode_id, stream_id, round(t_s, 6)))


def write_jsonl(path: str | Path, records: Iterable[VLMRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    {
                        "episode_id": record.episode_id,
                        "stream_id": record.stream_id,
                        "t_s": record.t_s,
                        "score": record.score,
                        "summary": record.summary,
                        "model": record.model,
                        "prompt_hash": record.prompt_hash,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
