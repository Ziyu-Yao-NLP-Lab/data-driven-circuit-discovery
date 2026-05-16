import pdb
import numpy as np
import argparse
import torch
from einops import einsum

from src.utils import general_utils as gu
from src.utils import metrics
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.utils import make_hooks_and_matrices, tokenize_plus

def parse_args():
    p = argparse.ArgumentParser(description="Faithfulness evaluation for circuits.")
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model name or short alias (e.g. gpt2, qwen2.5-7b)")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Model cache directory (e.g. ./models). Created if missing.")
    p.add_argument("--task", type=str, default="ioi",
                   help="Task name (used for result/data paths)")
    p.add_argument("--analyze_type", type=str, default="task-mixed",
                   help="Analysis type (e.g. used in output filename)")
    p.add_argument("--only_correct", type=str, default="correct",
                   choices=["correct", "both"],
                   help="Use only correct samples or both")
    p.add_argument("--circuit_files", type=str, nargs="+",
                   default=["2-operand-train-EAP-IG-inputs.pt", "3-operand-train-EAP-IG-inputs.pt"],
                   help="Circuit filenames (under results/circuits/<task>/<family>/<only_correct>/)")
    p.add_argument("--data_files", type=str, nargs="+",
                   default=["2-operand-test.csv", "3-operand-test.csv"],
                   help="Data filenames (under data/<task>/<family>/<only_correct>/)")
    return p.parse_args()

def compute_cpr_cmd(faithfulnesses, one_sided_cmd=True):
    """
    Compute CPR and CMD metrics from faithfulness scores at standard MIB thresholds.
    
    Args:
        faithfulnesses: dict mapping circuit size proportion (str) to faithfulness score
        one_sided_cmd: if True, CMD only penalizes when faithfulness < 1.0 (no penalty for overshooting)
    
    Returns:
        cpr: Circuit Performance Ratio (higher is better) - area under faithfulness curve
        cmd: Circuit-Model Distance (lower is better, 0 is best) - area between faithfulness curve and 1
    """
    percentages = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1)
    
    cpr = 0.0
    cmd = 0.0
    for i in range(len(percentages) - 1):
        x_1 = percentages[i]
        x_2 = percentages[i + 1]
 
        f1 = faithfulnesses[str(x_1)]
        f2 = faithfulnesses[str(x_2)]
 
        dx = x_2 - x_1
 
        # CPR: area under faithfulness curve
        cpr += dx * (f1 + f2) / 2
 
        # CMD: area between faithfulness curve and 1
        if one_sided_cmd:
            cmd += dx * (max(0, 1.0 - f1) + max(0, 1.0 - f2)) / 2
        else:
            cmd += dx * (abs(1.0 - f1) + abs(1.0 - f2)) / 2
 
    return cpr, cmd

def faithfulness_multi_circuit(args):
    """
    Evaluate faithfulness using multiple circuits, taking the max score per example.
    
    For each example, computes circuit_score from all circuits and takes the max.
    The overall circuit_score is the mean of these per-example max scores.
    """
    model_name = args.model_name
    cache_dir = args.cache_dir
    analyze_type = args.analyze_type
    only_correct = args.only_correct
    circuit_files = args.circuit_files
    data_files = args.data_files
    task = args.task
    
    model, tokenizer = gu.load_model(model_name, cache_dir=cache_dir)
    metric = metrics.get_logit_diff(model.tokenizer)

    # Set up paths based on task
    if task == "ioi":
        root_circuit_path = f"results/circuits/ioi/{gu.model2family(model_name)}/{only_correct}"
        root_data_path = f"data/ioi/{gu.model2family(model_name)}/{only_correct}"
    elif task == "arithmetic":
        root_circuit_path = f"results/circuits/arithmetic/{gu.model2family(model_name)}/{only_correct}"
        root_data_path = f"data/arithmetic/{gu.model2family(model_name)}/{only_correct}"
    elif task == "entity-binding":
        root_circuit_path = f"results/circuits/entity-binding/{gu.model2family(model_name)}/{only_correct}"
        root_data_path = f"data/entity-binding/{gu.model2family(model_name)}/{only_correct}"
    elif task == "mixed":
        root_circuit_path = f"results/circuits/mixed/{gu.model2family(model_name)}/{only_correct}"
        root_data_path = f"data/mixed/{gu.model2family(model_name)}/{only_correct}"
    else:
        raise ValueError(f"Invalid task: {task}")
    
    circuit_paths = [f"{root_circuit_path}/{f}" for f in circuit_files]
    data_paths = [f"{root_data_path}/{f}" for f in data_files]

    results = {}
    winner_tracking = {}  # Tracks which circuit won for each example

    for d_path in data_paths:
        print(f"Data: {d_path}")
        d_path_id = d_path.split("/")[-1].replace("-test.csv", "")
        results[d_path_id] = {}
        winner_tracking[d_path_id] = {}

        ds = gu.EAPDataset(d_path)
        dataloader = ds.to_dataloader(10)

        # Compute baseline score (full model)
        baseline_score = evaluate_baseline(model, dataloader, metric).mean().item()

        # Compute corrupted score (no edges) - use first circuit as reference for structure
        corrupted_circuit = Graph.from_json(circuit_paths[0]) if circuit_paths[0].endswith('.json') else Graph.from_pt(circuit_paths[0])
        corrupted_circuit.apply_topn(0, True)
        corrupted_score = evaluate_graph(model, corrupted_circuit, dataloader, metric).mean().item()

        results[d_path_id]['baseline_score'] = baseline_score
        results[d_path_id]['corrupted_score'] = corrupted_score
        results[d_path_id]['circuit_score'] = {}
        results[d_path_id]['faithfulness'] = {}

        for size in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
            size_str = str(size)
            winner_tracking[d_path_id][size_str] = []

            # Collect per-example scores from all circuits
            all_circuit_scores = []  # Shape will be (num_circuits, num_examples)
            
            for c_path in circuit_paths:
                circuit = Graph.from_json(c_path) if c_path.endswith('.json') else Graph.from_pt(c_path)
                top_n = int(len(circuit.edges) * size)
                circuit.apply_topn(top_n, absolute=True, prune=True)
                
                # Get per-example scores (don't call .mean() yet)
                per_example_scores = evaluate_graph(model, circuit, dataloader, metric)
                all_circuit_scores.append(per_example_scores)

            # Stack into (num_circuits, num_examples) tensor
            all_circuit_scores = np.stack([s.cpu().numpy() if hasattr(s, 'cpu') else np.array(s) for s in all_circuit_scores], axis=0)

            # Take max across circuits for each example
            max_scores = np.max(all_circuit_scores, axis=0)  # Shape: (num_examples,)
            winning_circuit_indices = np.argmax(all_circuit_scores, axis=0)  # Shape: (num_examples,)

            # Record winner tracking info for each example
            for example_idx in range(len(max_scores)):
                winner_info = {
                    "example_idx": int(example_idx),
                    "winning_circuit_idx": int(winning_circuit_indices[example_idx]),
                    "winning_circuit_file": circuit_files[winning_circuit_indices[example_idx]],
                    "all_scores": all_circuit_scores[:, example_idx].tolist()
                }
                winner_tracking[d_path_id][size_str].append(winner_info)

            # Compute overall circuit score as mean of per-example max scores
            circuit_score = float(np.mean(max_scores))
            results[d_path_id]['circuit_score'][size_str] = circuit_score

            # Compute faithfulness
            if baseline_score != corrupted_score:
                faithfulness_val = (circuit_score - corrupted_score) / (baseline_score - corrupted_score)
            else:
                faithfulness_val = 0.0
            results[d_path_id]['faithfulness'][size_str] = faithfulness_val

            print(f"data: {d_path}; circuit size: {size_str}; faithfulness: {faithfulness_val:.4f}")

    # Save results
    save_dir = f"results/circuit-eval/{task}/{gu.model2family(model_name)}/{only_correct}"
    results_filename = f"multi-circuit-{analyze_type}-analyze.json"
    winner_filename = f"multi-circuit-{analyze_type}-winner-tracking.json"
    
    gu.save_json_file(results, f"{save_dir}/{results_filename}")
    gu.save_json_file(winner_tracking, f"{save_dir}/{winner_filename}")

    print(f"Results saved to: {save_dir}/{results_filename}")
    print(f"Winner tracking saved to: {save_dir}/{winner_filename}")

    gu.clear_cache()

    return results, winner_tracking

def faithfulness(args):
    model_name = args.model_name
    cache_dir = args.cache_dir
    analyze_type = args.analyze_type
    only_correct = args.only_correct
    circuit_files = args.circuit_files
    data_files = args.data_files
    task = args.task
    model, tokenizer = gu.load_model(model_name, cache_dir=cache_dir)
    metric = metrics.get_logit_diff(model.tokenizer)

    circuit_paths = []
    data_paths = []
    if task == "ioi":
        root_circuit_path = f"results/circuits/ioi/{gu.model2family(model_name)}/{only_correct}"
        root_data_path = f"data/ioi/{gu.model2family(model_name)}/{only_correct}"
    elif task == "arithmetic":
        root_circuit_path = f"results/circuits/arithmetic/{gu.model2family(model_name)}/{only_correct}"
        root_data_path = f"data/arithmetic/{gu.model2family(model_name)}/{only_correct}"
    elif task == "entity-binding":
        root_circuit_path = f"results/circuits/entity-binding/{gu.model2family(model_name)}/{only_correct}"
        root_data_path = f"data/entity-binding/{gu.model2family(model_name)}/{only_correct}"
    elif task == "mixed":
        root_circuit_path = f"results/circuits/mixed/{gu.model2family(model_name)}/{only_correct}"
        root_data_path = f"data/mixed/{gu.model2family(model_name)}/{only_correct}"
    else:
        raise ValueError(f"Invalid task: {task}")
    circuit_paths.extend(f"{root_circuit_path}/{f}" for f in args.circuit_files)
    data_paths.extend(f"{root_data_path}/{f}" for f in args.data_files)

    for i, c_path in enumerate(circuit_paths):
        print(f"Circuit: {c_path}")
        circuit = Graph.from_json(c_path) if c_path.endswith('.json') else Graph.from_pt(c_path)
        
        results= {}
        for k, d_path in enumerate(data_paths):
            print(f"Data: {d_path}")
            d_path_id = d_path.split("/")[-1].replace("-test.csv", "")
            results[d_path_id] = {}
            ds = gu.EAPDataset(d_path)
            dataloader = ds.to_dataloader(10)
            baseline_score = evaluate_baseline(model, dataloader, metric).mean().item()

            corrupted_circuit = Graph.from_json(c_path) if c_path.endswith('.json') else Graph.from_pt(c_path)
            corrupted_circuit.apply_topn(0, True)

            corrupted_score = evaluate_graph(model, corrupted_circuit, dataloader, metric).mean().item()
            results[d_path_id][f'baseline_score'] = baseline_score
            results[d_path_id][f'corrupted_score'] = corrupted_score
            results[d_path_id][f'circuit_score'] = {}
            results[d_path_id]['faithfulness'] = {}
            for size in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
                circuit = Graph.from_json(c_path) if c_path.endswith('.json') else Graph.from_pt(c_path)
                top_n = int(len(circuit.edges)*size)
                circuit.apply_topn(top_n, absolute=True, prune=True)
                circuit_score = evaluate_graph(model, circuit, dataloader, metric).mean().item()
                size = str(size)
                results[d_path_id]["circuit_score"][size] = circuit_score
                if baseline_score != corrupted_score:
                    faithfulness = (circuit_score - corrupted_score) / (baseline_score - corrupted_score)
                else:
                    faithfulness = 0.0  
                results[d_path_id]['faithfulness'][size] = faithfulness
                print(f"circuit: {c_path}; data: {d_path}; circuit size: {size}; faithfulness: {faithfulness:.4f}")
        
        # save results
        save_path = c_path.split("/")[-1].replace(".pt", f"-{analyze_type}-analyze.json")
        gu.save_json_file(results, f"results/circuit-eval/{task}/{gu.model2family(model_name)}/{only_correct}/{save_path}")

        del circuit, results
        gu.clear_cache()

def run_faithfulness(model, circuit_path, data_path, metric, batch_size: int = 2):
    """Run the faithfulness experiment.

    Args:
        batch_size: DataLoader batch size for evaluate_graph/evaluate_baseline.
            Larger values speed up eval proportionally if GPU memory allows.
    """
    ds = gu.EAPDataset(data_path)
    dataloader = ds.to_dataloader(batch_size)
    with torch.no_grad():
        baseline_score = evaluate_baseline(model, dataloader, metric).mean().item()
    
    corrupted_circuit = Graph.from_json(circuit_path) if circuit_path.endswith('.json') else Graph.from_pt(circuit_path)
    corrupted_circuit.apply_topn(0, True)
    with torch.no_grad():
        corrupted_score = evaluate_graph(model, corrupted_circuit, dataloader, metric).mean().item()
    
    results = {
        'baseline_score': baseline_score,
        'corrupted_score': corrupted_score,
        'circuit_score': {},
        'faithfulness': {},
    }

    del corrupted_circuit
    gu.clear_cache()
    
    for size in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
        circuit = Graph.from_json(circuit_path) if circuit_path.endswith('.json') else Graph.from_pt(circuit_path)
        top_n = int(len(circuit.edges) * size)
        circuit.apply_topn(top_n, absolute=True, prune=True)
        with torch.no_grad():
            circuit_score = evaluate_graph(model, circuit, dataloader, metric).mean().item()
        
        size_str = str(size)
        results['circuit_score'][size_str] = circuit_score
        
        if baseline_score != corrupted_score:
            faithfulness = (circuit_score - corrupted_score) / (baseline_score - corrupted_score)
        else:
            faithfulness = 0.0
        results['faithfulness'][size_str] = faithfulness

        print(f"circuit: {circuit_path}; data: {data_path}; circuit size: {size_str}; faithfulness: {faithfulness:.4f}")
    del circuit
    gu.clear_cache()

    return results

def run_faithfulness_multi_circuit(model, circuit_paths, data_path, metric):
    """Optimized faithfulness eval that caches activation_difference per batch.

    The original eap evaluate_graph runs two forward passes per call (one
    corrupted to populate activation_difference, one masked clean to score
    the circuit). For multi-circuit faithfulness across many sizes, the
    corrupted pass produces an identical activation_difference for every
    (size, circuit) pair within a batch, so we run it once per batch and
    snapshot the resulting tensor. Each (size, circuit) then resets the
    cache and runs only the masked clean pass — roughly halving forward
    passes vs the v1_unbatched implementation.

    The pre-loaded circuits avoid re-reading and re-sorting graphs from disk
    on every (size, circuit) iteration as well.
    """
    if not (hasattr(model.cfg, "use_attn_result") and model.cfg.use_attn_result):
        raise AssertionError("Model must be configured with use_attn_result=True")
    if model.cfg.use_normalization_before_and_after:
        # The Gemma-style code path requires the separate-activations branch
        # of make_hooks_and_matrices, which we don't replicate here. Fall
        # back to the unbatched implementation in that case.
        return run_faithfulness_multi_circuit_v1_unbatched(
            model, circuit_paths, data_path, metric
        )

    sizes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    ds = gu.EAPDataset(data_path)
    dataloader = ds.to_dataloader(2)

    with torch.no_grad():
        baseline_score = evaluate_baseline(model, dataloader, metric).mean().item()

    # Use the first circuit's graph as the structural reference for hooks
    # and to get the total edge count for top-n calculations.
    first_path = circuit_paths[0]
    reference_graph = Graph.from_json(first_path) if first_path.endswith('.json') else Graph.from_pt(first_path)
    n_total_edges = len(reference_graph.edges)

    # Compute corrupted-circuit baseline once (size=0 → empty circuit).
    reference_graph.apply_topn(0, True)
    with torch.no_grad():
        corrupted_score = evaluate_graph(
            model, reference_graph, dataloader, metric
        ).mean().item()
    del reference_graph
    gu.clear_cache()

    # Pre-load all circuits and snapshot their original edge scores so we can
    # apply different top-n thresholds without re-reading from disk.
    raw_circuits = []
    for path in circuit_paths:
        g = Graph.from_json(path) if path.endswith('.json') else Graph.from_pt(path)
        raw_circuits.append(g)

    def build_circuit_for_size(graph, size):
        """Apply top-n thresholding for a given size, mutating the given graph."""
        top_n = int(n_total_edges * size)
        # apply_topn defaults to reset=True, so the in_graph mask is wiped
        # before the new top-n edges are selected.
        graph.apply_topn(top_n, absolute=True, prune=True)
        return graph

    # ---- Helpers ported from eap.evaluate.evaluate_graph (non-Gemma path) ----

    def make_input_construction_hook(activation_difference_buf, in_graph_vector):
        def input_construction_hook(activations, hook):
            update = einsum(
                activation_difference_buf[:, :, :len(in_graph_vector)],
                in_graph_vector,
                'batch pos previous hidden, previous ... -> batch pos ... hidden',
            )
            activations += update
            return activations
        return input_construction_hook

    def build_input_construction_hooks(graph, in_graph_matrix, activation_difference_buf):
        hooks = []
        for layer in range(model.cfg.n_layers):
            if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)) and \
                not all(parent_edge.in_graph
                        for head in range(model.cfg.n_heads)
                        for parent_edge in graph.nodes[f'a{layer}.h{head}'].parent_edges):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    hook = make_input_construction_hook(
                        activation_difference_buf,
                        in_graph_matrix[:prev_index, bwd_index],
                    )
                    hooks.append((node.qkv_inputs[i], hook))

            if graph.nodes[f'm{layer}'].in_graph and \
                not all(parent_edge.in_graph for parent_edge in graph.nodes[f'm{layer}'].parent_edges):
                node = graph.nodes[f'm{layer}']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                hook = make_input_construction_hook(
                    activation_difference_buf,
                    in_graph_matrix[:prev_index, bwd_index],
                )
                hooks.append((node.in_hook, hook))

        if not all(parent_edge.in_graph for parent_edge in graph.nodes['logits'].parent_edges):
            node = graph.nodes['logits']
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            hook = make_input_construction_hook(
                activation_difference_buf,
                in_graph_matrix[:fwd_index, bwd_index],
            )
            hooks.append((node.in_hook, hook))
        return hooks

    # ---- Per-batch loop with cached activation_difference ----

    # Accumulate per-example scores per size across batches.
    scores_per_size = {str(s): [] for s in sizes}

    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)
        batch_size = len(clean)

        # Build hooks against a fresh full-graph reference. Since hooks are
        # bound to forward_index/prev_index lookups that depend only on model
        # structure (not edge masks), we can use any graph here.
        reference = Graph.from_json(first_path) if first_path.endswith('.json') else Graph.from_pt(first_path)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(
            model, reference, batch_size, n_pos, None,
        )
        del reference

        # Step 1: corrupted forward pass — fills activation_difference with
        # corrupted activations at every source node.
        with torch.inference_mode():
            with model.hooks(fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

        # Snapshot so we can reset before each (size, circuit) iteration. The
        # clean pass below mutates activation_difference in-place by
        # subtracting clean activations as it runs.
        cached_corrupted_ad = activation_difference.clone()

        # Step 2: for each (size, circuit), reset AD and run the masked clean pass.
        for size in sizes:
            best_per_example = None
            for raw_graph in raw_circuits:
                circuit = build_circuit_for_size(raw_graph, size)

                in_graph_matrix = circuit.in_graph.to(
                    device=model.cfg.device, dtype=model.cfg.dtype,
                )
                in_graph_matrix = 1 - in_graph_matrix

                input_cons_hooks = build_input_construction_hooks(
                    circuit, in_graph_matrix, activation_difference,
                )

                # Reset activation_difference to the cached corrupted state
                # before each clean pass — fwd_hooks_clean will subtract clean
                # acts in-place during the pass, leaving (corrupted - clean).
                activation_difference.copy_(cached_corrupted_ad)

                with torch.inference_mode():
                    with model.hooks(fwd_hooks_clean + input_cons_hooks):
                        logits = model(clean_tokens, attention_mask=attention_mask)

                r = metric(logits, None, input_lengths, label).cpu()
                if r.dim() == 0:
                    r = r.unsqueeze(0)

                if best_per_example is None:
                    best_per_example = r
                else:
                    best_per_example = torch.maximum(best_per_example, r)

            scores_per_size[str(size)].append(best_per_example)

        # Free per-batch tensors before moving on.
        del activation_difference, cached_corrupted_ad
        gu.clear_cache()

    del raw_circuits
    gu.clear_cache()

    # Aggregate per-batch scores into a single per-size mean.
    results = {
        'baseline_score': baseline_score,
        'corrupted_score': corrupted_score,
        'circuit_score': {},
        'faithfulness': {},
    }
    for size in sizes:
        size_str = str(size)
        all_scores = torch.cat(scores_per_size[size_str])
        best_score = all_scores.mean().item()
        results['circuit_score'][size_str] = best_score
        if baseline_score != corrupted_score:
            faithfulness = (best_score - corrupted_score) / (baseline_score - corrupted_score)
        else:
            faithfulness = 0.0
        results['faithfulness'][size_str] = faithfulness
        print(f"data: {data_path}; circuit size: {size_str}; faithfulness: {faithfulness:.4f}")

    return results


def compute_per_example_faithfulness(
    model,
    circuit_paths,
    data_path,
    metric,
    sizes=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1),
    batch_size: int = 2,
):
    """Per-circuit per-example faithfulness across circuit sparsity thresholds.

    Wraps ``metric`` with ``mean=False`` so each batched evaluate_* call
    returns one score per example (default ``get_logit_diff`` would otherwise
    collapse each batch to a scalar). Loops over (circuit, size) and stores a
    full (k, n_sizes, n_test) tensor for downstream CPR/CMD analysis.

    Args:
        model: Loaded TransformerLens model.
        circuit_paths: List of .pt or .json circuit graphs to evaluate.
        data_path: Path to the test CSV.
        metric: Metric closure produced by metrics.get_logit_diff (or similar).
            Must accept a ``mean=False`` keyword to return per-example scores.
        sizes: Circuit sparsity thresholds (fraction of total edges).
        batch_size: Dataloader batch size.

    Returns:
        Dict with:
            faith:                 float32 ndarray (k, n_sizes, n_test)
            circuit_score:         float32 ndarray (k, n_sizes, n_test)
            sizes:                 float32 ndarray (n_sizes,)
            baseline_per_example:  float32 ndarray (n_test,)
            corrupted_per_example: float32 ndarray (n_test,)
    """
    def per_ex_metric(logits, clean_logits, input_length, labels):
        return metric(logits, clean_logits, input_length, labels, mean=False)

    ds = gu.EAPDataset(data_path)
    dataloader = ds.to_dataloader(batch_size)
    n_test = len(ds)

    with torch.no_grad():
        baseline_pe = evaluate_baseline(
            model, dataloader, per_ex_metric, quiet=True
        ).cpu().numpy()
    if len(baseline_pe) != n_test:
        raise RuntimeError(
            f"Per-example baseline returned {len(baseline_pe)} scores, expected "
            f"{n_test}. Confirm the metric supports mean=False."
        )

    first = circuit_paths[0]
    ref = Graph.from_json(first) if first.endswith('.json') else Graph.from_pt(first)
    ref.apply_topn(0, True)
    with torch.no_grad():
        corrupted_pe = evaluate_graph(
            model, ref, dataloader, per_ex_metric, quiet=True
        ).cpu().numpy()
    del ref
    gu.clear_cache()

    denom = baseline_pe - corrupted_pe
    denom_safe = np.where(np.abs(denom) < 1e-8, 1.0, denom)

    k = len(circuit_paths)
    n_sizes = len(sizes)
    circuit_score_arr = np.zeros((k, n_sizes, n_test), dtype=np.float32)
    faith_arr = np.zeros((k, n_sizes, n_test), dtype=np.float32)

    for i, cpath in enumerate(circuit_paths):
        for j, size in enumerate(sizes):
            circuit = (
                Graph.from_json(cpath) if cpath.endswith('.json')
                else Graph.from_pt(cpath)
            )
            top_n = int(len(circuit.edges) * size)
            circuit.apply_topn(top_n, absolute=True, prune=True)
            with torch.no_grad():
                scores = evaluate_graph(
                    model, circuit, dataloader, per_ex_metric, quiet=True
                ).cpu().numpy()
            if scores.shape[0] != n_test:
                raise RuntimeError(
                    f"Per-example scores shape {scores.shape} != ({n_test},) "
                    f"for circuit {cpath} at size {size}."
                )
            circuit_score_arr[i, j] = scores
            faith_arr[i, j] = (scores - corrupted_pe) / denom_safe
            del circuit
            gu.clear_cache()
        print(f"  [per-ex] {i + 1}/{k} circuits done")

    # Replace any NaN with 0 so downstream aggregation/plotting treats
    # numerically-unstable examples as "circuit contributes nothing"
    # rather than poisoning means and colormaps.
    np.nan_to_num(faith_arr, copy=False, nan=0.0)
    np.nan_to_num(circuit_score_arr, copy=False, nan=0.0)
    np.nan_to_num(baseline_pe, copy=False, nan=0.0)
    np.nan_to_num(corrupted_pe, copy=False, nan=0.0)

    return {
        'faith': faith_arr,
        'circuit_score': circuit_score_arr,
        'sizes': np.array(sizes, dtype=np.float32),
        'baseline_per_example': baseline_pe.astype(np.float32),
        'corrupted_per_example': corrupted_pe.astype(np.float32),
    }


def run_faithfulness_multi_circuit_v1_unbatched(model, circuit_paths, data_path, metric):
    """Original (slow) implementation. Kept as reference / fallback.

    Loops over (size, circuit) pairs and re-runs both clean and corrupted forward
    passes inside evaluate_graph each time. Replaced by run_faithfulness_multi_circuit
    which caches activation_difference per batch to halve forward passes.
    """
    ds = gu.EAPDataset(data_path)
    dataloader = ds.to_dataloader(2)
    with torch.no_grad():
        baseline_score = evaluate_baseline(model, dataloader, metric).mean().item()
    
    corrupted_circuit = Graph.from_json(circuit_paths[0]) if circuit_paths[0].endswith('.json') else Graph.from_pt(circuit_paths[0])
    corrupted_circuit.apply_topn(0, True)
    with torch.no_grad():
        corrupted_score = evaluate_graph(model, corrupted_circuit, dataloader, metric).mean().item()
    
    results = {
        'baseline_score': baseline_score,
        'corrupted_score': corrupted_score,
        'circuit_score': {},
        'faithfulness': {},
    }

    del corrupted_circuit
    gu.clear_cache()
    
    for size in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
        best_per_example = None
        
        for circuit_path in circuit_paths:
            circuit = Graph.from_json(circuit_path) if circuit_path.endswith('.json') else Graph.from_pt(circuit_path)
            top_n = int(len(circuit.edges) * size)
            circuit.apply_topn(top_n, absolute=True, prune=True)
            
            with torch.no_grad():
                scores = evaluate_graph(model, circuit, dataloader, metric)
            
            if best_per_example is None:
                best_per_example = scores
            else:
                best_per_example = torch.maximum(best_per_example, scores)
            
            del circuit  # free graph memory immediately
            gu.clear_cache()
        
        best_score = best_per_example.mean().item()
        
        size_str = str(size)
        results['circuit_score'][size_str] = best_score
        
        if baseline_score != corrupted_score:
            faithfulness = (best_score - corrupted_score) / (baseline_score - corrupted_score)
        else:
            faithfulness = 0.0
        results['faithfulness'][size_str] = faithfulness

        print(f"data: {data_path}; circuit size: {size_str}; faithfulness: {faithfulness:.4f}")
    
    gu.clear_cache()
    return results

def area_under_curve(faithfulnesses, log_scale:bool=False):
    percentages = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1)
    area_under = 0. # cpr
    area_from_1 = 0. # cmd
    for i in range(len(percentages)-1):
        i_1, i_2 = i, i+1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)

        trapezoidal = (x_2 - x_1) * \
                        (((abs(1. - faithfulnesses[str(x_1)])) + (abs(1. - faithfulnesses[str(x_2)]))) / 2)
        area_from_1 += trapezoidal 
        
        trapezoidal = (x_2 - x_1) * ((faithfulnesses[str(x_1)] + faithfulnesses[str(x_2)]) / 2)
        area_under += trapezoidal
    return area_under, area_from_1


def main():
    args = parse_args()
    model, tokenizer = gu.load_model(args.model_name, cache_dir=args.cache_dir)
    faithfulness(args)

if __name__ == "__main__":
    main()
    # Run from repo root. Example:
    # python -m src.circuit_discovery.faithfulness \
    #   --model_name meta-llama/Llama-3.1-8B --cache_dir ./models \
    #   --task ioi --circuit_files <c1.pt> <c2.pt> --data_files <d1.csv> <d2.csv>