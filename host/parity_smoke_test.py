from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


T = 8
INPUT_SHAPE = (T, 1, 2, 32, 32)
ARTIFACT = Path(__file__).parent / "artifacts" / "snn_smoke.onnx"


class ExplicitLIF(nn.Module):
    def __init__(self, decay: float = 0.5, threshold: float = 1.0) -> None:
        super().__init__()
        self.decay = decay
        self.threshold = threshold

    def forward(self, current: torch.Tensor, membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = self.decay * membrane + current
        spikes = (membrane >= self.threshold).to(current.dtype)
        membrane = membrane * (1.0 - spikes)
        return spikes, membrane


class SmokeSNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 3)
        self.lif = ExplicitLIF()

    def forward(self, event_sequence: torch.Tensor) -> torch.Tensor:
        membrane = torch.zeros(
            event_sequence.shape[1],
            4,
            event_sequence.shape[3] // 2,
            event_sequence.shape[4] // 2,
            dtype=event_sequence.dtype,
            device=event_sequence.device,
        )
        logits = []
        for time_index in range(T):
            current = self.encoder(event_sequence[time_index])
            spikes, membrane = self.lif(current, membrane)
            features = self.pool(spikes).flatten(1)
            logits.append(self.classifier(features))
        return torch.stack(logits).mean(dim=0)


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    model = SmokeSNN().eval()
    event_sequence = torch.rand(INPUT_SHAPE, dtype=torch.float32)

    with torch.no_grad():
        torch_output = model(event_sequence).cpu().numpy()

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (event_sequence,),
        ARTIFACT,
        input_names=["event_sequence"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )

    session = ort.InferenceSession(str(ARTIFACT), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["logits"], {"event_sequence": event_sequence.numpy()})[0]

    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5)
    print(f"PARITY_OK max_abs_error={np.max(np.abs(torch_output - onnx_output)):.8f}")
    print(f"MODEL={ARTIFACT}")


if __name__ == "__main__":
    main()
