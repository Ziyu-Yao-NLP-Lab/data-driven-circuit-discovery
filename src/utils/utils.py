from pathlib import Path
from typing import Union

import torch

from src.utils.general_utils import EAPDataset, tokenize_plus


def compute_dataset_accuracy(
    model,
    data_path: Union[str, Path],
    batch_size: int = 32,
) -> float:
    """Compute accuracy of the model on a dataset CSV.

    For each example the model is run on the clean prompt.  The prediction is
    correct when the logit for the correct token (labels[0]) is strictly
    greater than every incorrect token logit (labels[1:]).  Logits are read
    from the last non-padding token position.

    Args:
        model: Loaded TransformerLens HookedTransformer model.
        data_path: Path to a CSV with columns clean, corrupted, correct_idx,
            incorrect_idx (standard EAPDataset format).
        batch_size: Number of examples per forward pass.

    Returns:
        Accuracy as a float in [0, 1].
    """
    dataset = EAPDataset(str(data_path))
    dataloader = dataset.to_dataloader(batch_size)

    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for clean, _corrupted, labels in dataloader:
            tokens, _mask, input_lengths, _n_pos = tokenize_plus(model, clean)
            logits = model(tokens)  # (batch, seq, vocab)

            # Index the last real token for each example
            batch_size_actual = logits.size(0)
            batch_idx = torch.arange(batch_size_actual, device=logits.device)
            last_logits = logits[batch_idx, input_lengths - 1]  # (batch, vocab)

            # labels: (batch, n_candidates), first column is the correct token
            labels = labels.to(logits.device)
            for i in range(batch_size_actual):
                row = labels[i]
                row = row[row != 0]  # remove padding tokens
                cand_logits = last_logits[i, row]
                correct = cand_logits[0]
                incorrect_max = cand_logits[1:].max() if len(cand_logits) > 1 else torch.tensor(float("-inf"))
                if correct > incorrect_max:
                    n_correct += 1
                n_total += 1

    return n_correct / n_total if n_total > 0 else 0.0
