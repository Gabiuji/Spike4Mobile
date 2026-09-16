from pathlib import Path
import struct

import numpy as np

from aedat3 import load_gesture_window, read_gesture_labels


DATASET = Path(__file__).parent.parent / "data" / "dvsgesture128" / "DvsGesture" / "DvsGesture"
TRIAL = DATASET / "user01_natural.aedat"
LABELS = DATASET / "user01_natural_labels.csv"
OUTPUT_FILE = Path(__file__).parent / "artifacts" / "android_raw_sample.bin"
EXPECTED_FILE = Path(__file__).parent / "artifacts" / "android_raw_expected.bin"

HEADER = struct.Struct("<qqI")
EVENT = struct.Struct("<iiqB")


def main() -> None:
    label = read_gesture_labels(LABELS)[0]
    events = load_gesture_window(TRIAL, label)
    if events.size == 0:
        raise ValueError("selected gesture has no events")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("wb") as stream:
        stream.write(HEADER.pack(label.start_us, label.end_us, len(events)))
        for x, y, timestamp_us, polarity in events:
            stream.write(EVENT.pack(int(x), int(y), int(timestamp_us), int(polarity)))

    expected = np.zeros((8, 2, 32, 32), dtype=np.float32)
    duration = float(label.end_us - label.start_us)
    for x, y, timestamp_us, polarity in events:
        x //= 4
        y //= 4
        normalized_time = (timestamp_us - label.start_us) / duration * 7.0
        lower_bin = min(int(np.floor(normalized_time)), 7)
        upper_weight = normalized_time - lower_bin
        channel = 1 if polarity > 0 else 0
        expected[lower_bin, channel, y, x] += 1.0 - upper_weight
        if lower_bin + 1 < 8:
            expected[lower_bin + 1, channel, y, x] += upper_weight
    expected /= max(float(expected.sum()), 1.0)
    expected *= 1000.0
    with EXPECTED_FILE.open("wb") as stream:
        stream.write(expected.tobytes(order="C"))

    print(
        f"ANDROID_RAW_SAMPLE_OK events={len(events)} label={label.class_id - 1} "
        f"window=({label.start_us},{label.end_us}) output={OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
