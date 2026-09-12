from pathlib import Path

import numpy as np

from aedat3 import iter_polarity_events, load_gesture_window, read_gesture_labels


DATASET = (
    Path(__file__).parent.parent
    / "data"
    / "dvsgesture128"
    / "DvsGesture"
    / "DvsGesture"
)
TRIAL = DATASET / "user29_natural.aedat"
LABELS = DATASET / "user29_natural_labels.csv"


def main() -> None:
    labels = read_gesture_labels(LABELS)
    assert labels, "expected annotated gestures"

    first_events = []
    for event in iter_polarity_events(TRIAL):
        first_events.append(event)
        if len(first_events) == 1000:
            break

    events = np.asarray(first_events, dtype=np.int64)
    assert events.shape == (1000, 4)
    assert np.all((events[:, 0] >= 0) & (events[:, 0] < 128))
    assert np.all((events[:, 1] >= 0) & (events[:, 1] < 128))
    assert set(np.unique(events[:, 3])).issubset({0, 1})
    assert np.all(events[1:, 2] >= events[:-1, 2])

    gesture = load_gesture_window(TRIAL, labels[0], max_events=1000)
    assert gesture.shape[1] == 4
    assert gesture.shape[0] > 0
    assert np.all(gesture[:, 2] >= labels[0].start_us)
    assert np.all(gesture[:, 2] <= labels[0].end_us)
    print(
        f"AEDAT_OK events={len(events)} first_gesture_events={len(gesture)} "
        f"labels={len(labels)} sensor=128x128"
    )


if __name__ == "__main__":
    main()
