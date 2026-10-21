"""Train the S2 CNN on user-supplied positive and negative RNA sequences."""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data import MAX_LEN, VOCAB_MAP, load_sample_records
from main_CNN import CNN1DModel, GeneSequenceDataset

PROJECT_DIR = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def select_device(requested="auto"):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def environment_info(device):
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("torch", "numpy", "pandas", "scikit-learn")},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-file", "--positive_file", type=Path, required=True)
    parser.add_argument("--negative-file", "--negative_file", type=Path, required=True)
    parser.add_argument("--indices-file", "--indices_file", type=Path,
                        help="Optional saved split JSON; otherwise create a stratified 80/20 split")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--lr-patience", type=int, default=7)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", "--output_dir", type=Path,
                        default=PROJECT_DIR / "outputs" / "single_run")
    parser.add_argument("--checkpoint", type=Path,
                        help="Weight output path; defaults to OUTPUT_DIR/best_model.pt")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or args.early_stop_patience < 1:
        parser.error("epochs and early-stop-patience must be positive; batch-size must be >= 2")
    if args.learning_rate <= 0 or not 0 < args.lr_factor < 1 or args.min_lr < 0:
        parser.error("invalid learning-rate schedule")
    if not 0 <= args.threshold <= 1 or args.num_workers < 0:
        parser.error("threshold must be in [0, 1]; num-workers must be nonnegative")
    return args


def write_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(args):
    records = load_sample_records(args.positive_file, args.negative_file, negative_sample_ratio=1.0)
    sequences = records["Sequence_around_stop"].tolist()
    labels = records["label"].to_numpy(dtype=np.float32)
    if args.indices_file is None:
        train_idx, val_idx = train_test_split(
            np.arange(len(records)), test_size=0.2, random_state=42, stratify=labels)
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    else:
        if not args.indices_file.is_file():
            raise FileNotFoundError(f"Split file not found: {args.indices_file}")
        with args.indices_file.open(encoding="utf-8") as handle:
            indices = json.load(handle)
        try:
            train_idx = np.asarray(indices["train_set"]["all_indices"])
            val_idx = np.asarray(indices["validation_set"]["all_indices"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Split JSON must contain train_set.all_indices and validation_set.all_indices") from exc
        if any(part.ndim != 1 or not np.issubdtype(part.dtype, np.integer)
               for part in (train_idx, val_idx)):
            raise ValueError("Split indices must be one-dimensional lists of integers")
    all_idx = np.concatenate([train_idx, val_idx])
    if not np.array_equal(np.sort(all_idx), np.arange(len(records))):
        raise ValueError("Split indices must partition every sample exactly once, without overlap")
    if len(np.unique(labels[train_idx])) != 2 or len(np.unique(labels[val_idx])) != 2:
        raise ValueError("Training and validation sets must each contain positive and negative samples")
    return sequences, labels, dict(VOCAB_MAP), MAX_LEN, train_idx, val_idx, records


def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    total_loss, targets, probs = 0.0, [], []
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs, batch_targets = inputs.to(device), batch_targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, batch_targets)
            total_loss += loss.item() * inputs.size(0)
            probs.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
            targets.extend(batch_targets.cpu().numpy().reshape(-1).tolist())
    targets, probs = np.asarray(targets, dtype=np.float32), np.asarray(probs, dtype=np.float32)
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    return {
        "loss": float(total_loss / max(1, len(loader.dataset))),
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "auroc": float(roc_auc_score(targets, probs)),
        "average_precision": float(average_precision_score(targets, probs)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "targets": targets, "probs": probs, "preds": preds,
    }


def train(args):
    set_seed(args.seed)
    device = select_device(args.device)
    sequences, labels, vocab_map, max_len, train_idx, val_idx, records = load_dataset(args)
    if len(train_idx) % args.batch_size == 1:
        raise ValueError("This batch size leaves one training sample; BatchNorm requires >= 2 per batch")
    output_dir = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else output_dir / "best_model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_dataset = GeneSequenceDataset([sequences[int(i)] for i in train_idx], labels[train_idx], vocab_map, max_len)
    val_sequences = [sequences[int(i)] for i in val_idx]
    val_dataset = GeneSequenceDataset(val_sequences, labels[val_idx], vocab_map, max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pos_weight = float(np.sum(labels[train_idx] == 0) / np.sum(labels[train_idx] == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    model = CNN1DModel(vocab_size=len(vocab_map), max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor,
                                  patience=args.lr_patience, min_lr=args.min_lr)
    best_auc, best_epoch, best_state, no_improve, history = -1.0, 0, None, 0, []
    start_time = time.time()
    print(f"Seed {args.seed}; device {device}; checkpoint policy: best validation AUROC")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        for inputs, batch_targets in train_loader:
            inputs, batch_targets = inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * inputs.size(0)
        val_metrics = evaluate(model, val_loader, criterion, device, args.threshold)
        scheduler.step(val_metrics["auroc"])
        history.append({
            "epoch": epoch, "train_loss": train_loss_total / max(1, len(train_dataset)),
            "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
            "val_auroc": val_metrics["auroc"], "val_average_precision": val_metrics["average_precision"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(f"Epoch {epoch:03d} val_auc={val_metrics['auroc']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_precision={val_metrics['precision']:.4f} "
              f"val_recall={val_metrics['recall']:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if val_metrics["auroc"] > best_auc:
            best_auc, best_epoch = val_metrics["auroc"], epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, criterion, device, args.threshold)
    # Store a plain CPU state_dict so users can load it with weights_only=True.
    torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, checkpoint)
    reloaded = CNN1DModel(vocab_size=len(vocab_map), max_len=max_len).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    reloaded_metrics = evaluate(reloaded, val_loader, criterion, device, args.threshold)
    if not np.array_equal(final_metrics["preds"], reloaded_metrics["preds"]) or not np.allclose(
            final_metrics["probs"], reloaded_metrics["probs"], rtol=1e-6, atol=1e-7):
        raise RuntimeError("Checkpoint reload did not preserve validation predictions")
    config = {key: value for key, value in vars(args).items()
              if key not in ("output_dir", "checkpoint", "positive_file", "negative_file", "indices_file")}
    config.update({"positive_file": Path(args.positive_file).name,
                   "negative_file": Path(args.negative_file).name,
                   "indices_file": args.indices_file.name if args.indices_file else None,
                   "vocab_map": vocab_map, "max_len": max_len,
                   "hidden_units": 256, "hidden_dropout": 0.5,
                   "checkpoint_policy": "highest_validation_auroc_first_tie"})
    summary = {
        "experiment_config": config, "environment": environment_info(device),
        "train_size": len(train_idx), "validation_size": len(val_idx),
        "train_positive": int(labels[train_idx].sum()),
        "train_negative": int(len(train_idx) - labels[train_idx].sum()),
        "validation_positive": int(labels[val_idx].sum()),
        "validation_negative": int(len(val_idx) - labels[val_idx].sum()), "pos_weight": pos_weight,
        "best_epoch": best_epoch, "epochs_run": len(history), "best_validation_auroc": best_auc,
        "seconds": time.time() - start_time, "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_reload_verified": True,
        "metrics": {key: value for key, value in final_metrics.items() if key not in ("targets", "probs", "preds")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "split.json").write_text(json.dumps({
        "train_set": {"all_indices": train_idx.tolist()},
        "validation_set": {"all_indices": val_idx.tolist()},
    }, indent=2) + "\n", encoding="utf-8")
    write_rows(output_dir / "history.csv", history)
    write_rows(output_dir / "metrics.csv", [summary["metrics"]])
    write_rows(output_dir / "predictions.csv", [
        {"saved_dataset_index": int(val_idx[i]), "Name": records.iloc[int(val_idx[i])]["Name"],
         "Sequence_around_stop": val_sequences[i], "target": int(final_metrics["targets"][i]),
         "probability": float(final_metrics["probs"][i]), "prediction": int(final_metrics["preds"][i])}
        for i in range(len(val_idx))
    ])
    print(json.dumps(summary["metrics"], sort_keys=True))
    return summary


def main():
    args = parse_args()
    try:
        train(args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
