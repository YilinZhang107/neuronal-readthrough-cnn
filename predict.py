"""Predict RT probabilities from a 200-nt RNA TSV and a released checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MAX_LEN, VOCAB_MAP, read_sequence_table
from main_CNN import CNN1DModel, GeneSequenceDataset
from train import select_device


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = CNN1DModel(vocab_size=4, max_len=MAX_LEN)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"Cannot load {checkpoint.name} as an S2 CNN state_dict: {exc}") from exc
    return model.to(device).eval()


def predict_frame(frame, model, device, batch_size=32):
    dataset = GeneSequenceDataset(frame["Sequence_around_stop"].tolist(),
                                  np.zeros(len(frame), dtype=np.float32), dict(VOCAB_MAP), MAX_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    with torch.inference_mode():
        for inputs, _ in loader:
            probabilities.extend(torch.sigmoid(model(inputs.to(device))).cpu().numpy().reshape(-1).tolist())
    output = frame.copy()
    output["rt_probability"] = probabilities
    output["predicted_label"] = (output["rt_probability"] > 0.5).astype(int)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="TSV with Name and Sequence_around_stop")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.output.resolve() in (args.input.resolve(), args.checkpoint.resolve()):
        parser.error("output must differ from the input and checkpoint files")
    try:
        frame = read_sequence_table(args.input)
        device = select_device(args.device)
        model = load_model(args.checkpoint, device)
        result = predict_frame(frame, model, device, args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Predicted {len(result)} sequences on {device}; wrote {args.output}")


if __name__ == "__main__":
    main()
