from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


_HEADER = struct.Struct("<HHIIIIII")
_EVENT = struct.Struct("<II")
_POLARITY_EVENT_TYPES = {1}


@dataclass(frozen=True)
class GestureLabel:
    class_id: int
    start_us: int
    end_us: int


def read_gesture_labels(path: str | Path) -> list[GestureLabel]:
    labels = np.genfromtxt(path, delimiter=",", names=True, dtype=np.int64)
    if labels.size == 0:
        return []
    if labels.ndim == 0:
        labels = np.array([labels], dtype=labels.dtype)
    return [
        GestureLabel(int(row[0]), int(row[1]), int(row[2]))
        for row in labels.tolist()
    ]


def iter_polarity_events(path: str | Path):
    """Yield AEDAT 3.1 polarity events as x, y, timestamp_us, polarity."""
    with Path(path).open("rb") as stream:
        # Locate the first binary polarity packet after the textual preamble.
        packet_prefix = struct.pack("<HHI", 1, 1, _EVENT.size)
        preamble = stream.read(4096)
        data_start = preamble.find(packet_prefix)
        if data_start < 0:
            raise ValueError("AEDAT polarity packet header not found")
        stream.seek(data_start)

        while True:
            header = stream.read(_HEADER.size)
            if not header:
                return
            if len(header) != _HEADER.size:
                raise ValueError("truncated AEDAT event header")

            event_type, _, event_size, _, timestamp_overflow, _, event_number, _ = _HEADER.unpack(header)
            if event_size < _EVENT.size:
                raise ValueError(f"unsupported AEDAT event size: {event_size}")

            for _ in range(event_number):
                raw_event = stream.read(event_size)
                if len(raw_event) != event_size:
                    raise ValueError("truncated AEDAT event block")
                if event_type not in _POLARITY_EVENT_TYPES:
                    continue

                data, timestamp = _EVENT.unpack_from(raw_event)
                x = (data >> 17) & 0x1FFF
                y = (data >> 2) & 0x1FFF
                polarity = (data >> 1) & 0x1
                timestamp_us = (timestamp_overflow << 32) + timestamp
                yield x, y, timestamp_us, polarity


def load_gesture_window(
    event_path: str | Path,
    label: GestureLabel,
    *,
    max_events: int | None = None,
) -> np.ndarray:
    """Load one annotated gesture as an Nx4 int64 array."""
    selected = []
    for event in iter_polarity_events(event_path):
        if label.start_us <= event[2] <= label.end_us:
            selected.append(event)
            if max_events is not None and len(selected) >= max_events:
                break
    return np.asarray(selected, dtype=np.int64).reshape((-1, 4))


def load_gesture_windows(
    event_path: str | Path,
    labels: list[GestureLabel],
) -> list[np.ndarray]:
    """Read one AEDAT stream once and return events grouped by annotation."""
    windows = [[] for _ in labels]
    label_index = 0
    for event in iter_polarity_events(event_path):
        timestamp_us = event[2]
        while label_index < len(labels) and timestamp_us > labels[label_index].end_us:
            label_index += 1
        if label_index >= len(labels):
            break
        label = labels[label_index]
        if timestamp_us >= label.start_us:
            windows[label_index].append(event)

    return [np.asarray(window, dtype=np.int64).reshape((-1, 4)) for window in windows]
