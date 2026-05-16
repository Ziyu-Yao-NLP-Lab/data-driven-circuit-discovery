"""Fault-tolerant monkey-patch for eap.attribute.get_scores_eap_ig.

The upstream eap library raises ValueError on the first NaN it sees during
EAP-IG attribution.  In practice, a handful of examples in the sequence-completion
3-gram-symbolic data produce NaN gradients (likely bf16 layer-norm / softmax
instability on heavily-repeated token sequences), and losing ~45 min of
attribution work per bad example is untenable.

This patch snapshots `scores` before each batch, runs the IG loop, and if a
NaN is detected at any point it *restores* the snapshot and skips the batch.
Normalization uses the number of successful batches rather than the total.

Usage (call once, before `eap.attribute.attribute` is invoked):

    from src.utils import eap_patch
    eap_patch.apply()
"""

from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from tqdm import tqdm

import eap.attribute as eap_attribute_module
from eap.utils import tokenize_plus, make_hooks_and_matrices
from eap.graph import Graph


_original_get_scores_eap_ig: Optional[Callable] = None


def get_scores_eap_ig_safe(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps: int = 30,
    quiet: bool = False,
):
    """Drop-in replacement for eap.attribute.get_scores_eap_ig that tolerates NaN batches."""
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward),
        device="cuda",
        dtype=model.cfg.dtype,
    )

    total_items = 0
    skipped_items = 0

    iterator = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in iterator:
        batch_size = len(clean)
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, n_pos_corrupted = tokenize_plus(model, corrupted)

        if n_pos != n_pos_corrupted:
            print(
                f"Position mismatch ({n_pos} vs {n_pos_corrupted}); skipping batch. "
                f"clean[0]={clean[0][:60]!r}"
            )
            skipped_items += batch_size
            continue

        # Snapshot scores before this batch so we can undo any NaN contamination.
        scores_snapshot = scores.clone()
        batch_failed = False
        failure_reason = ""

        try:
            (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
                make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
            )

            with torch.inference_mode():
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
                input_activations_corrupted = activation_difference[
                    :, :, graph.forward_index(graph.nodes["input"])
                ].clone()

                with model.hooks(fwd_hooks=fwd_hooks_clean):
                    clean_logits = model(clean_tokens, attention_mask=attention_mask)
                input_activations_clean = (
                    input_activations_corrupted
                    - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]
                )

            def input_interpolation_hook(k: int):
                def hook_fn(activations, hook):
                    new_input = input_activations_corrupted + (k / steps) * (
                        input_activations_clean - input_activations_corrupted
                    )
                    new_input.requires_grad = True
                    return new_input

                return hook_fn

            for step in range(0, steps):
                with model.hooks(
                    fwd_hooks=[
                        (
                            graph.nodes["input"].out_hook,
                            input_interpolation_hook(step),
                        )
                    ],
                    bwd_hooks=bwd_hooks,
                ):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)
                    if torch.isnan(metric_value).any().item():
                        batch_failed = True
                        failure_reason = f"metric NaN at step {step}"
                        break
                    metric_value.backward()

                if torch.isnan(scores).any().item():
                    batch_failed = True
                    failure_reason = f"scores NaN after step {step} backward"
                    break

        except Exception as e:
            batch_failed = True
            failure_reason = f"exception: {type(e).__name__}: {e}"

        if batch_failed:
            scores.copy_(scores_snapshot)
            skipped_items += batch_size
            print(
                f"WARNING: NaN batch skipped ({failure_reason}). "
                f"clean[0]={clean[0][:60]!r}"
            )
        else:
            total_items += batch_size

    if total_items == 0:
        raise RuntimeError(
            f"All {skipped_items} examples failed with NaN during EAP-IG attribution. "
            "This indicates a systemic numerical problem, not sporadic glitches."
        )

    scores /= total_items
    scores /= steps

    print(
        f"EAP-IG attribution complete: {total_items} successful, "
        f"{skipped_items} skipped (NaN)."
    )
    return scores


def apply() -> None:
    """Monkey-patch eap.attribute.get_scores_eap_ig with the NaN-tolerant version."""
    global _original_get_scores_eap_ig
    if _original_get_scores_eap_ig is not None:
        return
    _original_get_scores_eap_ig = eap_attribute_module.get_scores_eap_ig
    eap_attribute_module.get_scores_eap_ig = get_scores_eap_ig_safe
    print("[eap_patch] Installed NaN-tolerant get_scores_eap_ig.")
