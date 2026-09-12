from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from train_snn import ARTIFACT, TrainableSNN, build_samples


ONNX_ARTIFACT = Path(__file__).parent / "artifacts" / "snn_gesture_trained.onnx"


def main() -> None:
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    model = TrainableSNN(bins=checkpoint["bins"], classes=checkpoint["classes"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    inputs, _ = build_samples(
        "trials_to_train.txt",
        max_trials=5,
        max_gestures=1,
        bins=checkpoint["bins"],
        cache_dir=Path(__file__).parent / "artifacts" / "sample_cache",
    )
    sample = inputs[:1].to(dtype=torch.float32)

    with torch.no_grad():
        torch_output = model(sample).numpy()

    ONNX_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        ONNX_ARTIFACT,
        input_names=["event_sequence"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
    )

    session = ort.InferenceSession(str(ONNX_ARTIFACT), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["logits"], {"event_sequence": sample.numpy()})[0]
    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5)
    predicted_class = int(np.argmax(onnx_output[0]))
    max_error = float(np.max(np.abs(torch_output - onnx_output)))
    print(
        f"EXPORT_OK input={tuple(sample.shape)} output={tuple(onnx_output.shape)} "
        f"predicted_class={predicted_class} max_abs_error={max_error:.8f} "
        f"model={ONNX_ARTIFACT}"
    )


if __name__ == "__main__":
    main()
