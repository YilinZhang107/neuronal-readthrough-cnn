"""Original balanced TXT-data workflow with explicit sample identities."""
from pathlib import Path

import numpy as np
import pandas as pd

VOCAB_MAP = {"A": 0, "C": 1, "G": 2, "U": 3}
MAX_LEN = 200
SEQUENCE_COLUMN = "Sequence_around_stop"


def read_sequence_table(path, sequence_column_name=SEQUENCE_COLUMN):
    """Read a TSV, preserving row order and rejecting malformed model inputs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    try:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Sequence file is empty: {path}") from exc
    required = {"Name", sequence_column_name}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path.name}: missing TSV columns {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError(f"{path.name}: no input sequences")
    frame = frame.copy()
    frame["Name"] = frame["Name"].str.strip()
    frame[sequence_column_name] = frame[sequence_column_name].str.strip().str.upper()
    for index, row in frame.iterrows():
        name, sequence = row["Name"], row[sequence_column_name]
        location = f"{path.name}, record {index + 1} ({name or 'unnamed'})"
        if not name:
            raise ValueError(f"{location}: Name must not be empty")
        if len(sequence) != MAX_LEN:
            raise ValueError(f"{location}: expected {MAX_LEN} nt, got {len(sequence)}")
        invalid = sorted(set(sequence) - set(VOCAB_MAP))
        if invalid:
            raise ValueError(f"{location}: only ACGU is accepted; invalid characters: {invalid}")
    return frame


def load_sample_records(positive_file_path, negative_file_path,
                        sequence_column_name=SEQUENCE_COLUMN, negative_sample_ratio=1.0):
    """Keep the original positive-first and pandas negative-sampling order."""
    positive = read_sequence_table(positive_file_path, sequence_column_name)
    negative = read_sequence_table(negative_file_path, sequence_column_name)
    positive["source_row"] = positive.index
    negative["source_row"] = negative.index
    positive["source_file"] = Path(positive_file_path).name
    negative["source_file"] = Path(negative_file_path).name
    positive["label"] = 1
    negative["label"] = 0
    if negative_sample_ratio is not None:
        count = int(len(positive) * negative_sample_ratio)
        if count <= 0:
            raise ValueError("negative_sample_ratio must select at least one negative sample")
        if len(negative) > count:
            negative = negative.sample(n=count, random_state=42)
    records = pd.concat([positive, negative], ignore_index=True)
    records.insert(0, "dataset_index", np.arange(len(records)))
    return records


def load_data_from_txt_files(positive_file_path, negative_file_path,
                             sequence_column_name=SEQUENCE_COLUMN,
                             negative_sample_ratio=None):
    """Retain the original loader's public return value and sampling behavior."""
    records = load_sample_records(positive_file_path, negative_file_path,
                                  sequence_column_name, negative_sample_ratio)
    sequences = records[sequence_column_name].tolist()
    labels = records["label"].to_numpy(dtype=np.float32)
    found_chars = set("".join(sequences))
    return sequences, labels, found_chars
