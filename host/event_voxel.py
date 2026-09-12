from pathlib import Path

import numpy as np


EventArray = np.ndarray


def load_events_csv(path: str | Path) -> EventArray:
    """Load rows with x,y,timestamp_us,polarity into an Nx4 integer array."""
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.int64,
        encoding="utf-8",
    )
    if data.size == 0:
        return np.empty((0, 4), dtype=np.int64)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    columns = ("x", "y", "timestamp_us", "polarity")
    missing = [column for column in columns if column not in data.dtype.names]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
    return np.column_stack([data[column] for column in columns]).astype(np.int64, copy=False)


def build_voxel_grid(
    events: EventArray,
    *,
    width: int,
    height: int,
    bins: int,
    window_start_us: int,
    window_end_us: int,
) -> np.ndarray:
    """Build a [bins, polarity, height, width] linearly interpolated voxel grid."""
    if width <= 0 or height <= 0 or bins <= 0:
        raise ValueError("width, height, and bins must be positive")
    if window_end_us <= window_start_us:
        raise ValueError("window_end_us must be greater than window_start_us")
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N, 4]: x, y, timestamp_us, polarity")

    grid = np.zeros((bins, 2, height, width), dtype=np.float32)
    duration = float(window_end_us - window_start_us)

    for x, y, timestamp_us, polarity in events:
        if timestamp_us < window_start_us or timestamp_us > window_end_us:
            continue
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"event coordinate outside sensor: ({x}, {y})")

        normalized_time = (timestamp_us - window_start_us) / duration * (bins - 1)
        lower_bin = min(int(np.floor(normalized_time)), bins - 1)
        upper_weight = normalized_time - lower_bin
        polarity_channel = 1 if polarity > 0 else 0
        grid[lower_bin, polarity_channel, y, x] += 1.0 - upper_weight

        upper_bin = lower_bin + 1
        if upper_bin < bins:
            grid[upper_bin, polarity_channel, y, x] += upper_weight

    return grid
