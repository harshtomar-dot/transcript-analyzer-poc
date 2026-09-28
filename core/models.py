from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional


@dataclass
class Turn:
    call_id: str
    index: int
    speaker: str  # "agent" or "user"
    text: str

    @property
    def key(self) -> str:
        return f"{self.call_id}::{self.index}"


@dataclass
class Call:
    call_id: str
    turns: list = field(default_factory=list)  # list[Turn]
    date: Optional[_date] = None
    duration_seconds: Optional[float] = None
    source_filename: Optional[str] = None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def context_for(self, index: int, window: int = 2) -> str:
        start = max(0, index - window)
        lines = []
        for t in self.turns[start:index]:
            lines.append(f"{t.speaker}: {t.text}")
        return "\n".join(lines)
