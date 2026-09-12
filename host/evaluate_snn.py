import argparse
from pathlib import Path

import torch

from train_snn import ARTIFACT, TrainableSNN, build_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SNN on DVS Gesture test trials")
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--max-gestures", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    model = TrainableSNN(bins=checkpoint["bins"], classes=checkpoint["classes"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    cache_dir = Path(__file__).parent / "artifacts" / "sample_cache"
    inputs, targets = build_samples(
        "trials_to_test.txt",
        args.max_trials,
        args.max_gestures,
        bins=checkpoint["bins"],
        cache_dir=cache_dir,
    )

    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    print(f"EVAL_OK samples={len(targets)} accuracy={accuracy:.3f} checkpoint={Path(ARTIFACT)}")


if __name__ == "__main__":
    main()
