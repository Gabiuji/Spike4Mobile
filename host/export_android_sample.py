from pathlib import Path

import numpy as np


CACHE_FILE = Path(__file__).parent / "artifacts" / "sample_cache" / "trials_to_train_user01_natural_8.npz"
OUTPUT_FILE = Path(__file__).parent / "artifacts" / "android_input.bin"
LABEL_FILE = Path(__file__).parent / "artifacts" / "android_input_label.txt"


def main() -> None:
    cached = np.load(CACHE_FILE)
    sample = np.asarray(cached["inputs"][0], dtype=np.float32)
    label = int(cached["targets"][0])
    expected_shape = (8, 2, 32, 32)
    if sample.shape != expected_shape:
        raise ValueError(f"expected sample shape {expected_shape}, got {sample.shape}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.tofile(OUTPUT_FILE)
    LABEL_FILE.write_text(f"{label}\n", encoding="ascii")
    print(
        f"ANDROID_SAMPLE_OK shape={sample.shape} label={label} "
        f"min={sample.min():.6f} max={sample.max():.6f} file={OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
