# Translational readthrough CNN

CNN code and five pretrained model weights for the paper:

**Neuronal stop-codon readthrough is associated with ribosome pausing and appends lineage-restricted extensions that can alter protein localization in Drosophila**

The model follows Figure S2. Data are not included; supply your own RNA sequences.

## Install

Use Python 3.12 and run commands from this directory:

```bash
python -m pip install -r requirements.txt
```

## Input format

Use a tab-separated file (`.tsv`) with these columns:

| Column | Content |
| --- | --- |
| `Name` | Sequence identifier |
| `Sequence_around_stop` | Exactly 200 RNA nucleotides using A, C, G and U |

Each sequence contains 100 nt upstream of the stop codon, the stop codon, and 97 nt downstream. The stop codon occupies zero-based indices 100–102. Lowercase letters are converted to uppercase; invalid bases or lengths are rejected.

## Load weights and predict

```bash
python predict.py --input your_sequences.tsv --checkpoint saved_model/seed_9.pt --output predictions.csv --device cpu
```

Any of the five `.pt` files in `saved_model/` can be used. `--device auto` uses CUDA when available. The output retains the input columns and adds `rt_probability` and `predicted_label` (`1` when probability is greater than 0.5, otherwise `0`).

To load a checkpoint directly in Python:

```python
import torch
from main_CNN import CNN1DModel

model = CNN1DModel(vocab_size=4, max_len=200)
model.load_state_dict(torch.load("saved_model/seed_9.pt", map_location="cpu", weights_only=True))
model.eval()
# x: a float32 one-hot tensor of shape [N, 4, 200], channels A, C, G, U.
# with torch.no_grad():
#     probabilities = torch.sigmoid(model(x)).squeeze(1)
```

## Train on your own data

Provide separate positive and negative TSV files in the same format:

```bash
python train.py --positive-file positives.tsv --negative-file negatives.tsv --seed 9 --output-dir outputs/train
```

Excess negative samples are downsampled to the positive count. A stratified 80/20 training/validation split uses `random_state=42`. Training saves the best validation-AUROC weights to `outputs/train/best_model.pt`. The saved split can be reused with `--indices-file outputs/train/split.json` on the same input data.

Code is available under the [MIT License](LICENSE).
