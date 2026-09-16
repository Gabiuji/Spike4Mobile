from pathlib import Path

import numpy as np


CACHE_DIR = Path(__file__).parent / "artifacts" / "sample_cache"
OUTPUT_FILE = Path(__file__).parent / "artifacts" / "android_test_inputs.bin"
LABEL_FILE = Path(__file__).parent / "artifacts" / "android_test_labels.bin"


def main() -> None:
    inputs = []
    targets = []
    for path in sorted(CACHE_DIR.glob("trials_to_test_*.npz")):
        cached = np.load(path)
        inputs.append(cached["inputs"])
        targets.append(cached["targets"])

    samples = np.concatenate(inputs, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(targets, axis=0).astype(np.int32, copy=False)
    samples = samples[:30]
    labels = labels[:30]
    if samples.shape != (30, 8, 2, 32, 32) or labels.shape != (30,):
        raise ValueError(f"unexpected batch shapes: inputs={samples.shape}, labels={labels.shape}")

    masses = samples.sum(axis=(1, 2, 3, 4), keepdims=True)
    samples = samples / np.maximum(masses, 1.0) * 1000.0
    samples.tofile(OUTPUT_FILE)
    labels.tofile(LABEL_FILE)
    print(
        f"ANDROID_TEST_BATCH_OK samples={len(labels)} shape={samples.shape} "
        f"labels={labels.tolist()} files={OUTPUT_FILE},{LABEL_FILE}"
    )


if __name__ == "__main__":
    main()
