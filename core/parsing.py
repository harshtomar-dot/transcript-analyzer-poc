import json
import re
from datetime import datetime
from typing import Optional

from .models import Call, Turn

AGENT_LABELS = {"agent", "bot", "assistant", "ai"}
USER_LABELS = {"user", "customer", "caller", "human", "lead"}

_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z _-]*)\s*:\s*(.*)$")


def _normalize_speaker(label: str) -> Optional[str]:
    label = label.strip().lower()
    if label in AGENT_LABELS:
        return "agent"
    if label in USER_LABELS:
        return "user"
    return None


def _parse_date(value) -> Optional[object]:
    if not value:
        return None
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(value[: len(fmt) + 2].split(".")[0], fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def parse_json_call(raw: dict, fallback_id: str, fallback_filename: Optional[str] = None) -> Call:
    call_id = str(raw.get("call_id") or raw.get("id") or fallback_id)
    date = _parse_date(raw.get("date") or raw.get("created_at") or raw.get("timestamp"))
    duration = raw.get("duration_seconds") or raw.get("duration") or raw.get("call_duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None

    raw_turns = raw.get("turns") or raw.get("messages") or raw.get("transcript") or []
    turns = []
    for i, t in enumerate(raw_turns):
        speaker_raw = str(t.get("speaker") or t.get("role") or t.get("from") or "").strip().lower()
        speaker = _normalize_speaker(speaker_raw) or ("agent" if speaker_raw in ("assistant",) else "user")
        text = str(t.get("text") or t.get("message") or t.get("content") or "").strip()
        if not text:
            continue
        turns.append(Turn(call_id=call_id, index=i, speaker=speaker, text=text))

    return Call(call_id=call_id, turns=turns, date=date, duration_seconds=duration, source_filename=fallback_filename)


def parse_text_call(raw_text: str, fallback_id: str, fallback_filename: Optional[str] = None) -> Call:
    turns = []
    current_speaker = None
    current_lines: list = []
    index = 0

    def flush():
        nonlocal current_speaker, current_lines, index
        if current_speaker and current_lines:
            text = " ".join(l.strip() for l in current_lines if l.strip())
            if text:
                turns.append(Turn(call_id=fallback_id, index=index, speaker=current_speaker, text=text))
                index += 1
        current_lines = []

    for line in raw_text.splitlines():
        m = _LINE_RE.match(line)
        if m:
            speaker = _normalize_speaker(m.group(1))
            if speaker:
                flush()
                current_speaker = speaker
                current_lines = [m.group(2)]
                continue
        if current_speaker:
            current_lines.append(line)
    flush()

    return Call(call_id=fallback_id, turns=turns, source_filename=fallback_filename)


def parse_uploaded_bytes(content: bytes, filename: str, fallback_id: str) -> Call:
    text = content.decode("utf-8", errors="replace")
    if filename.lower().endswith(".json"):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{filename}: invalid JSON ({e})")
        if isinstance(raw, list):
            raw = {"turns": raw}
        return parse_json_call(raw, fallback_id=fallback_id, fallback_filename=filename)
    return parse_text_call(text, fallback_id=fallback_id, fallback_filename=filename)
