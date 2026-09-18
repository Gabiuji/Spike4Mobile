import argparse
from pathlib import Path

import torch

from train_snn import ARTIFACT, TrainableSNN, build_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SNN on DVS Gesture test trials")
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--max-gestures", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    model = TrainableSNN(bins=checkpoint["bins"], classes=checkpoint["classes"]).to(device)
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
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    confusion = torch.zeros((checkpoint["classes"], checkpoint["classes"]), dtype=torch.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[target, prediction] += 1
    class_totals = confusion.sum(dim=1)
    class_hits = confusion.diag()
    class_accuracy = torch.where(
        class_totals > 0,
        class_hits.float() / class_totals.float(),
        torch.zeros_like(class_hits, dtype=torch.float32),
    )
    print(
        f"EVAL_OK samples={len(targets)} accuracy={accuracy:.3f} "
        f"device={device} checkpoint={Path(ARTIFACT)}"
    )
    print(f"CONFUSION_MATRIX rows=target columns=prediction\n{confusion.tolist()}")
    print(
        "CLASS_ACCURACY "
        + " ".join(f"{index + 1}:{value:.3f}" for index, value in enumerate(class_accuracy.tolist()))
    )


if __name__ == "__main__":
    main()
