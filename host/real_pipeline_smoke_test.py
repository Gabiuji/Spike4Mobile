from pathlib import Path

import numpy as np
import onnxruntime as ort

from aedat3 import load_gesture_window, read_gesture_labels
from event_voxel import build_voxel_grid


ROOT = Path(__file__).parent
DATASET = ROOT.parent / "data" / "dvsgesture128" / "DvsGesture" / "DvsGesture"
TRIAL = DATASET / "user29_natural.aedat"
LABELS = DATASET / "user29_natural_labels.csv"
MODEL = ROOT / "artifacts" / "snn_smoke.onnx"


def main() -> None:
    labels = read_gesture_labels(LABELS)
    first_label = labels[0]
    events = load_gesture_window(TRIAL, first_label, max_events=200_000)
    if events.size == 0:
        raise RuntimeError("first annotated gesture contains no events")

    # The smoke model uses 32x32; aggregate the 128x128 sensor by 4x4 cells.
    events_32 = events.copy()
    events_32[:, 0] //= 4
    events_32[:, 1] //= 4
    voxel_grid = build_voxel_grid(
        events_32,
        width=32,
        height=32,
        bins=8,
        window_start_us=first_label.start_us,
        window_end_us=first_label.end_us,
    )
    model_input = voxel_grid[:, np.newaxis, :, :, :].astype(np.float32)

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    output = session.run(["logits"], {"event_sequence": model_input})[0]
    predicted_class = int(np.argmax(output[0]))

    assert model_input.shape == (8, 1, 2, 32, 32)
    assert output.shape == (1, 3)
    print(
        f"PIPELINE_OK events={len(events)} input={model_input.shape} "
        f"output={output.shape} predicted_class={predicted_class} "
        f"label={first_label.class_id}"
    )


if __name__ == "__main__":
    main()
