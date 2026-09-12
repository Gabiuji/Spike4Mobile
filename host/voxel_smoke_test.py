from pathlib import Path

import numpy as np

from event_voxel import build_voxel_grid, load_events_csv


ROOT = Path(__file__).parent
SAMPLE_CSV = ROOT / "artifacts" / "events_sample.csv"


def main() -> None:
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_CSV.write_text(
        "x,y,timestamp_us,polarity\n"
        "1,2,0,1\n"
        "1,2,50,-1\n"
        "1,2,100,1\n",
        encoding="utf-8",
    )

    events = load_events_csv(SAMPLE_CSV)
    grid = build_voxel_grid(
        events,
        width=4,
        height=4,
        bins=4,
        window_start_us=0,
        window_end_us=100,
    )

    np.testing.assert_equal(grid.shape, (4, 2, 4, 4))
    np.testing.assert_allclose(grid[0, 1, 2, 1], 1.0)
    np.testing.assert_allclose(grid[1, 0, 2, 1], 0.5)
    np.testing.assert_allclose(grid[2, 0, 2, 1], 0.5)
    np.testing.assert_allclose(grid[3, 1, 2, 1], 1.0)
    np.testing.assert_allclose(grid.sum(), 3.0)
    print(f"VOXEL_OK shape={grid.shape} total={grid.sum():.1f}")


if __name__ == "__main__":
    main()
