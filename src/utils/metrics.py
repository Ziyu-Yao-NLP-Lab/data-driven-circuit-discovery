import pdb
import torch

from jaxtyping import Bool, Float, Int
from torch import Tensor

def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def get_incorrect_prob_mass(tokenizer):
    def incorrect(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,  # each label: [correct_id, wrong_id1, ...]
        mean=True,
        loss=False
    ):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)

        results = []
        for idx, label in enumerate(labels):
            label = label[label != 0].tolist() # remove labels with value zero (used for padding)
            cand_probs = probs[idx, label]
            p_correct = cand_probs[0] # first index is correct label
            p_incorrect = cand_probs[1:].sum()
            p_incorrect_max = cand_probs[1:].max()
            results.append(p_incorrect - p_incorrect_max) # remove the max incorrect prob mass

        results = torch.stack(results)
        if loss:
            results = -results  # EAP-IG convention
        if mean:
            results = results.mean()
        return results

    return incorrect

def get_normalized_logit(tokenizer):
    # Optional: candidate subset if you want to restrict vocab
    # entity_ids = torch.tensor([...])  # e.g., token ids of Ann, Joe, Pete, Tim

    def nl(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,  # each label: [correct_id, wrong_id1, ...]
        mean=True,
        loss=False
    ):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)

        results = []
        for idx, label in enumerate(labels):
            label = label[label != 0].tolist() # remove labels with value zero (used for padding)
            max_prob = probs[idx].max()
            cand_probs = probs[idx, label]
            p_correct = cand_probs[0] # first index is correct label
            results.append(max_prob/p_correct)

        results = torch.stack(results)
        if loss:
            results = -results  # EAP-IG convention
        if mean:
            results = results.mean()
        return results

    return nl

def get_logit_diff_w_sum(tokenizer):
    def logit_diff(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,  # each label: [correct_id, wrong_id1, ...]
        mean=True,
        loss=False
    ):
        logits = get_logit_positions(logits, input_length)

        results = []
        for idx, label in enumerate(labels):
            label = label[label != 0].tolist() # remove labels with value zero (used for padding)
            cand_logits = logits[idx, label]
            logit_correct = cand_logits[0] # first index is correct label
            logit_incorrect = cand_logits[1:].sum()
            logit_diff = logit_correct - logit_incorrect
            
            results.append(logit_diff)

        results = torch.stack(results)
        if loss:
            results = -results  # EAP-IG convention
        if mean:
            results = results.mean()
        
        if torch.isnan(results).any().item():
            # optional debug prints
            print("metric is NaN in metrics")
        return results

    return logit_diff


def logits_to_logit_diff_single(logits, answer_tokens):
    """
    Calculates logit difference for a single example.
    
    Args:
        logits: Shape [seq_len, vocab_size] 
        answer_tokens: Shape [2] containing [correct_token_id, incorrect_token_id]
    """
    # 1. Remove batch dimension if it exists (e.g., shape [1, 15, 50257])
    if logits.ndim == 3:
        logits = logits[0]  # Now shape is [15, 50257]
        
    final_logits = logits[-1, :]
    
    # 2. Extract the specific logits for the answer tokens
    # We don't need .gather() for a single 1D vector, we can just index directly
    correct_token_id = answer_tokens[0]
    incorrect_token_id = answer_tokens[1]
    
    correct_logit = final_logits[correct_token_id]
    incorrect_logit = final_logits[incorrect_token_id]
    
    # 3. Return the difference
    return correct_logit - incorrect_logit

def ioi_metric_single(
    logits,
    answer_tokens,
    corrupted_logit_diff,
    clean_logit_diff,
):
    """
    Calculates the IOI metric for a single example.
    """
    # Calculate the logit difference for this specific run
    patched_logit_diff = logits_to_logit_diff_single(logits, answer_tokens)
    
    # Apply the normalization formula
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def logits_to_ave_logit_diff(
    logits,
    answer_tokens,
    per_prompt: bool = False,
):
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def ioi_metric(
    logits,
    answer_tokens,
    corrupted_logit_diff,
    clean_logit_diff,
) :
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is same as on
    corrupted input, and 1 when performance is same as on clean input.
    """
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def faithfulness_metric(
    patched_logit_diff,
    corrupted_logit_diff,
    clean_logit_diff
) :
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is same as on
    corrupted input, and 1 when performance is same as on clean input.
    """
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def get_accuracy(tokenizer):
    def logit_diff(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,  # each label: [correct_id, wrong_id1, ...]
        mean=True,
        loss=False
    ):
        logits = get_logit_positions(logits, input_length)

        results = []
        for idx, label_row in enumerate(labels):

            label_row = label_row[label_row != 0].tolist() # remove labels with value zero (used for padding)
            cand_logits = logits[idx, label_row]

            logit_correct = cand_logits[0] # first index is correct label
            logit_incorrect = cand_logits[1:].max()
            if logit_correct > logit_incorrect:
                logit_diff = 1
            else:
                logit_diff = 0
            # print(f"idx: {idx}, logit_correct: {logit_correct}, logit_incorrect: {logit_incorrect}, logit_diff: {logit_diff}")
            results.append(logit_diff)
        # print(f"---"*20)
        # print(results)
        # results = torch.stack(results)
        if loss:
            results = -results  # EAP-IG convention
        if mean:
            results = sum(results) / len(results)
        
        return torch.tensor(results)
    return logit_diff

def get_logit_diff(tokenizer):
    """Build a logit_diff metric closure with a per-call NaN counter.

    Numerical instability in heavily-pruned circuits (especially with
    bf16/fp16 layer norms and softmax) can occasionally make per-example
    logit_diff NaN. Rather than letting NaN poison aggregation
    (`torch.maximum`, `mean`, `compute_cpr_cmd`), we replace per-example
    NaN values with 0.0 — a "this example contributes nothing" signal —
    and bookkeep how many examples were affected.

    The closure exposes:
        logit_diff.nan_count    : int, total examples replaced
        logit_diff.eval_count   : int, total examples evaluated
        logit_diff.reset_counts(): clear both counters
        logit_diff.NAN_HARD_LIMIT: hard cap; raises if exceeded in one call
    """
    def logit_diff(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,  # each label: [correct_id, wrong_id1, ...]
        mean=True,
        loss=False
    ):
        logits = get_logit_positions(logits, input_length)

        results = []
        for idx, label_row in enumerate(labels):

            label_row = label_row[label_row != 0].tolist() # remove labels with value zero (used for padding)
            cand_logits = logits[idx, label_row]

            logit_correct = cand_logits[0] # first index is correct label
            logit_incorrect = cand_logits[1:].max()
            logit_diff_val = logit_correct - logit_incorrect
            results.append(logit_diff_val)

        results = torch.stack(results)
        if loss:
            results = -results  # EAP-IG convention

        # Replace per-example NaN with 0 and track how many we hit.
        nan_mask = torch.isnan(results)
        n_nan = int(nan_mask.sum().item())
        logit_diff.eval_count += int(results.numel())
        if n_nan > 0:
            logit_diff.nan_count += n_nan
            if n_nan > logit_diff.NAN_HARD_LIMIT:
                raise ValueError(
                    f"logit_diff: {n_nan} NaN examples in a single call "
                    f"(> NAN_HARD_LIMIT={logit_diff.NAN_HARD_LIMIT}). "
                    "This indicates a systemic numerical problem, not a sporadic glitch."
                )
            print(
                f"WARNING: logit_diff: replacing {n_nan} NaN example(s) with 0 "
                f"(running total: {logit_diff.nan_count}/{logit_diff.eval_count})"
            )
            results = torch.where(nan_mask, torch.zeros_like(results), results)

        if mean:
            results = results.mean()
        return results

    logit_diff.nan_count = 0
    logit_diff.eval_count = 0
    logit_diff.NAN_HARD_LIMIT = 10

    def reset_counts():
        logit_diff.nan_count = 0
        logit_diff.eval_count = 0
    logit_diff.reset_counts = reset_counts

    return logit_diff

def get_entropy(tokenizer):
    # Optional: candidate subset if you want to restrict vocab
    # entity_ids = torch.tensor([...])  # e.g., token ids of Ann, Joe, Pete, Tim

    def entropy(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,  # each label: [correct_id, wrong_id1, ...]
        mean=True,
        loss=False
    ):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)

        results = []
        for idx, label in enumerate(labels):
            label = label[label != 0].tolist() # remove labels with value zero (used for padding)
            cand_probs = probs[idx, label]
            p_correct = cand_probs[0] # first index is correct label
            p_incorrect = cand_probs[1:]
            # entropy calculation of p_incorrect
            entropy_incorrect = -torch.sum(p_incorrect * torch.log(p_incorrect + 1e-10))
            results.append(entropy_incorrect)

        results = torch.stack(results)
        if loss:
            results = -results  # EAP-IG convention
        if mean:
            results = results.mean()
        return results

    return entropy


def get_prob_diff_v1(tokenizer):
    # Optional: candidate subset if you want to restrict vocab
    # entity_ids = torch.tensor([...])  # e.g., token ids of Ann, Joe, Pete, Tim

    def prob_diff(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,  # each label: [correct_id, wrong_id1, ...]
        mean=True,
        loss=False
    ):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)

        results = []
        for idx, label in enumerate(labels):
            label = label[label != 0].tolist() # remove labels with value zero (used for padding)
            cand_probs = probs[idx, label]
            p_correct = cand_probs[0] # first index is correct label
            p_incorrect = cand_probs[1:].sum()
            results.append(p_correct - p_incorrect)

        results = torch.stack(results)
        if loss:
            results = -results  # EAP-IG convention
        if mean:
            results = results.mean()
        return results

    return prob_diff


def get_prob_diff_v2(tokenizer):
    # Optional: candidate subset if you want to restrict vocab
    # entity_ids = torch.tensor([...])  # e.g., token ids of Ann, Joe, Pete, Tim

    def prob_diff(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,  # each label: [correct_id, wrong_id1, ...]
        mean=True,
        loss=False
    ):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)

        results = []
        for idx, label in enumerate(labels):
            label = label[label != 0].tolist() # remove labels with value zero (used for padding)
            cand_probs = probs[idx, label]
            p_correct = cand_probs[0] # first index is correct label
            # Find the largest value among incorrect labels
            p_incorrect = cand_probs[1:].max()
            results.append(p_correct - p_incorrect)

        results = torch.stack(results)
        if loss:
            results = -results  # EAP-IG convention
        if mean:
            results = results.mean()
        return results

    return prob_diff


def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

def accuracy(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True):
    logits = get_logit_positions(logits, input_length)
    predicted_token = logits.argmax(dim=-1).squeeze()

    correct = (predicted_token == labels[:, 0].to(logits.device)).float()
    pdb.set_trace() 
    print(f"{adf}")
    
    return correct