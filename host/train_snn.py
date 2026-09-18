import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aedat3 import load_gesture_windows, read_gesture_labels
from event_voxel import build_voxel_grid


DATASET = (
    Path(__file__).parent.parent
    / "data"
    / "dvsgesture128"
    / "DvsGesture"
    / "DvsGesture"
)
ARTIFACT = Path(__file__).parent / "artifacts" / "snn_gesture_checkpoint.pt"


class SurrogateSpike(torch.autograd.Function):
    threshold = 0.75

    @staticmethod
    def forward(context, membrane: torch.Tensor) -> torch.Tensor:
        context.save_for_backward(membrane)
        return (membrane >= SurrogateSpike.threshold).to(membrane.dtype)

    @staticmethod
    def backward(context, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (membrane,) = context.saved_tensors
        slope = 5.0
        surrogate_gradient = slope / (
            1.0 + (slope * (membrane - SurrogateSpike.threshold)).abs()
        ).square()
        return gradient * surrogate_gradient,


class TrainableSNN(nn.Module):
    def __init__(self, bins: int = 8, classes: int = 11) -> None:
        super().__init__()
        self.bins = bins
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, classes)
        self.decay = 0.5

    def forward(self, event_sequence: torch.Tensor) -> torch.Tensor:
        membrane = torch.zeros(
            event_sequence.shape[0],
            16,
            event_sequence.shape[3] // 2,
            event_sequence.shape[4] // 2,
            dtype=event_sequence.dtype,
            device=event_sequence.device,
        )
        logits = []
        for time_index in range(self.bins):
            current = self.encoder(event_sequence[:, time_index])
            membrane = self.decay * membrane + current
            spikes = SurrogateSpike.apply(membrane)
            membrane = membrane * (1.0 - spikes)
            spike_features = self.pool(spikes).flatten(1)
            membrane_features = self.pool(membrane).flatten(1)
            logits.append(self.classifier(torch.cat((spike_features, membrane_features), dim=1)))
        return torch.stack(logits, dim=1).mean(dim=1)


def build_samples(
    split_file: str,
    max_trials: int,
    max_gestures: int,
    bins: int,
    cache_dir: Path | None = None,
    balance_classes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    trial_names = [
        line.strip()
        for line in (DATASET / split_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:max_trials]
    samples = []
    targets = []
    for trial_name in trial_names:
        trial_path = DATASET / trial_name
        labels_path = trial_path.with_name(f"{trial_path.stem}_labels.csv")
        labels = read_gesture_labels(labels_path)
        labels_needed = min(len(labels), max_gestures - len(samples))
        selected_labels = labels[:labels_needed]
        trial_cache = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            trial_cache = cache_dir / (
                f"{Path(split_file).stem}_{trial_path.stem}_{bins}_{labels_needed}.npz"
            )
        legacy_cache = None
        if cache_dir is not None:
            legacy_cache = cache_dir / f"{Path(split_file).stem}_{trial_path.stem}_{bins}.npz"

        if trial_cache is not None and trial_cache.exists():
            print(f"CACHE_HIT trial={trial_name}", flush=True)
            cached = np.load(trial_cache)
            trial_inputs = cached["inputs"]
            trial_targets = cached["targets"]
        elif legacy_cache is not None and legacy_cache.exists():
            print(f"CACHE_HIT_LEGACY trial={trial_name}", flush=True)
            cached = np.load(legacy_cache)
            trial_inputs = cached["inputs"][:labels_needed]
            trial_targets = cached["targets"][:labels_needed]
        else:
            print(f"CACHE_BUILD trial={trial_name}", flush=True)
            gesture_windows = load_gesture_windows(trial_path, selected_labels)
            trial_inputs = []
            trial_targets = []
            for label, events in zip(selected_labels, gesture_windows):
                if events.size == 0:
                    continue
                events[:, 0] //= 4
                events[:, 1] //= 4
                trial_inputs.append(
                    build_voxel_grid(
                        events,
                        width=32,
                        height=32,
                        bins=bins,
                        window_start_us=label.start_us,
                        window_end_us=label.end_us,
                    )
                )
                trial_targets.append(label.class_id - 1)
            trial_inputs = np.asarray(trial_inputs, dtype=np.float32)
            trial_targets = np.asarray(trial_targets, dtype=np.int64)
            if trial_cache is not None:
                np.savez_compressed(trial_cache, inputs=trial_inputs, targets=trial_targets)
                print(f"CACHE_SAVED trial={trial_name} samples={len(trial_targets)}", flush=True)

        for grid, target in zip(trial_inputs, trial_targets):
            samples.append(grid)
            targets.append(int(target))
            if len(samples) >= max_gestures:
                break
        if len(samples) >= max_gestures:
            break

    if not samples:
        raise RuntimeError("no training samples were created")
    inputs = np.stack(samples).astype(np.float32, copy=False)
    sample_mass = inputs.sum(axis=(1, 2, 3, 4), keepdims=True)
    inputs = np.divide(inputs, np.maximum(sample_mass, 1.0), dtype=np.float32) * 1000.0
    targets_array = np.asarray(targets, dtype=np.int64)
    if balance_classes:
        class_indices = []
        class_counts = np.bincount(targets_array, minlength=11)
        samples_per_class = int(class_counts[class_counts > 0].min())
        for class_id in range(len(class_counts)):
            indices = np.flatnonzero(targets_array == class_id)[:samples_per_class]
            class_indices.extend(indices.tolist())
        class_indices = np.asarray(sorted(class_indices), dtype=np.int64)
        inputs = inputs[class_indices]
        targets_array = targets_array[class_indices]
    return torch.from_numpy(inputs), torch.from_numpy(targets_array)


def main() -> None:
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train a small SNN on annotated DVS Gesture windows")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=2)
    parser.add_argument("--max-gestures", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    cache_dir = Path(__file__).parent / "artifacts" / "sample_cache"
    inputs, targets = build_samples(
        "trials_to_train.txt",
        args.max_trials,
        args.max_gestures,
        bins=8,
        cache_dir=cache_dir,
        balance_classes=True,
    )
    model = TrainableSNN().to(device).train()
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        print(
            f"epoch={epoch} samples={len(targets)} loss={loss.item():.4f} "
            f"accuracy={accuracy:.3f} device={device}"
        )

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "bins": 8, "classes": 11}, ARTIFACT)
    elapsed_seconds = time.perf_counter() - started_at
    print(
        f"TRAIN_OK samples={len(targets)} elapsed_seconds={elapsed_seconds:.1f} "
        f"artifact={ARTIFACT}"
    )


if __name__ == "__main__":
    main()
