from typing import Optional


def filter_calls(
    calls: list,
    date_range: Optional[tuple] = None,
    duration_range: Optional[tuple] = None,
    turn_count_range: Optional[tuple] = None,
) -> list:
    result = []
    for call in calls:
        if date_range and call.date is not None:
            if not (date_range[0] <= call.date <= date_range[1]):
                continue
        if duration_range and call.duration_seconds is not None:
            if not (duration_range[0] <= call.duration_seconds <= duration_range[1]):
                continue
        if turn_count_range:
            if not (turn_count_range[0] <= call.turn_count <= turn_count_range[1]):
                continue
        result.append(call)
    return result
