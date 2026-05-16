"""Evaluate edge overlap and cross-dataset faithfulness between variant circuits.

For each attribution method, load all variant circuits discovered in step 02
and compute:
  1. Pairwise edge-overlap matrix (Jaccard on top-gamma edges, gamma=0.99).
  2. Pairwise cross-dataset faithfulness matrix. Entry [A][B] is the
     faithfulness of circuit A when evaluated on variant B's test set, using
     run_faithfulness (MIB CPR-style curve over 10 sizes). We collapse the
     curve into a single scalar via area-under-the-faithfulness-curve (CPR).

Low off-diagonal values in both matrices = circuits are dataset-specific.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from omegaconf import OmegaConf

from src.circuit_discovery.faithfulness import run_faithfulness, compute_cpr_cmd
from src.dcd.clustering import sparsify_binary
from src.utils import metrics
from src.utils.general_utils import load_model, model2family
from src.utils.graph_utils import build_edge_index, load_scores_fast

GAMMA = 0.99


def parse_list(flag_value: Optional[str]) -> Optional[List[str]]:
    if flag_value is None:
        return None
    return [x.strip() for x in flag_value.split(",") if x.strip()]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two binary edge masks."""
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return intersection / union


def compute_edge_overlap_matrix(circuit_paths: Dict[str, str], gamma: float) -> Dict[str, Dict[str, float]]:
    """Load each circuit, binarize with gamma, compute pairwise Jaccard."""
    variant_names = list(circuit_paths.keys())
    ref_path = circuit_paths[variant_names[0]]
    _, rows, cols = build_edge_index(ref_path)

    binaries: Dict[str, np.ndarray] = {}
    for variant, path in circuit_paths.items():
        scores = load_scores_fast(path, rows, cols)
        binaries[variant] = sparsify_binary(scores, gamma)

    matrix: Dict[str, Dict[str, float]] = {}
    for a in variant_names:
        matrix[a] = {}
        for b in variant_names:
            matrix[a][b] = jaccard(binaries[a], binaries[b])
    return matrix


def compute_cross_faithfulness_matrix(
    model,
    circuit_paths: Dict[str, str],
    test_paths: Dict[str, str],
    eval_batch_size: int = 2,
) -> Dict[str, Dict[str, dict]]:
    """For each (circuit variant A, data variant B), run run_faithfulness(A, B)."""
    metric = metrics.get_logit_diff(model.tokenizer)
    results: Dict[str, Dict[str, dict]] = {}
    for a, circuit_path in circuit_paths.items():
        results[a] = {}
        for b, data_path in test_paths.items():
            print(f"  faithfulness: circuit={a} | data={b}")
            r = run_faithfulness(model, circuit_path, data_path, metric,
                                 batch_size=eval_batch_size)
            cpr, cmd = compute_cpr_cmd(r["faithfulness"])
            results[a][b] = {
                "faithfulness_curve": r["faithfulness"],
                "circuit_score_curve": r["circuit_score"],
                "baseline_score": r["baseline_score"],
                "corrupted_score": r["corrupted_score"],
                "cpr": cpr,
                "cmd": cmd,
            }
    return results


def write_summary_tables(
    method: str,
    overlap: Dict[str, Dict[str, float]],
    faithfulness: Dict[str, Dict[str, dict]],
    out_dir: str,
) -> None:
    variants = list(overlap.keys())
    rows = []
    for i, a in enumerate(variants):
        for b in variants[i + 1 :]:
            rows.append(
                {
                    "method": method,
                    "variant_A": a,
                    "variant_B": b,
                    "edge_overlap": overlap[a][b],
                    "faith_A_on_B": faithfulness[a][b]["faithfulness_curve"]["0.05"],
                    "faith_B_on_A": faithfulness[b][a]["faithfulness_curve"]["0.05"],
                }
            )

    csv_path = os.path.join(out_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "variant_A",
                "variant_B",
                "edge_overlap",
                "faith_A_on_B",
                "faith_B_on_A",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = os.path.join(out_dir, "summary_table.txt")
    with open(txt_path, "w") as f:
        f.write(
            f"{'method':<16}{'variant_A':<28}{'variant_B':<28}"
            f"{'edge_overlap':<16}{'faith_A_on_B':<16}{'faith_B_on_A':<16}\n"
        )
        f.write("-" * 120 + "\n")
        for row in rows:
            f.write(
                f"{row['method']:<16}{row['variant_A']:<28}{row['variant_B']:<28}"
                f"{row['edge_overlap']:<16.4f}{row['faith_A_on_B']:<16.4f}{row['faith_B_on_A']:<16.4f}\n"
            )
    print(f"  wrote {csv_path}")
    print(f"  wrote {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method names to evaluate (subset of config methods)",
    )
    parser.add_argument(
        "--aspects",
        type=str,
        default=None,
        help="Comma-separated aspect names to evaluate (subset of config.data.aspects)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluation even if outputs already exist",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="DataLoader batch size used inside run_faithfulness. Larger "
             "values speed up evaluation if GPU memory allows.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    task = config.data.task_name
    model_family = model2family(config.model.name)

    method_filter = parse_list(args.methods)
    all_methods = [m.name for m in config.attribution.methods]
    methods = [m for m in all_methods if method_filter is None or m in method_filter]
    if not methods:
        print("No methods selected — check --methods.")
        return

    results_root = f"results/general-task-circuits/{task}/{model_family}"
    circuits_root = os.path.join(results_root, "circuits")
    eval_root = os.path.join(results_root, "evaluation")
    os.makedirs(eval_root, exist_ok=True)
    shutil.copy(args.config, results_root)

    variants = [v.prompt_type for v in config.data.variants]

    aspects_cfg = OmegaConf.to_container(config.data.get("aspects", {}), resolve=True)
    if not aspects_cfg:
        raise ValueError(
            "config.data.aspects is required — specify at least one aspect "
            "(e.g., complexity/domain/syntax) mapping to its variant list."
        )
    aspect_filter = parse_list(args.aspects)
    if aspect_filter is not None:
        aspects_cfg = {a: v for a, v in aspects_cfg.items() if a in aspect_filter}
        if not aspects_cfg:
            print("No aspects selected — check --aspects.")
            return

    # Validate aspect variants against the flat variants list.
    variant_set = set(variants)
    for aspect, aspect_variants in aspects_cfg.items():
        unknown = [v for v in aspect_variants if v not in variant_set]
        if unknown:
            raise ValueError(
                f"Aspect '{aspect}' references unknown variants {unknown} — "
                f"must be a subset of config.data.variants."
            )

    # Test data paths come from the same layout used by 01_create_data.py.
    test_paths: Dict[str, str] = {}
    for prompt_type in variants:
        test_path = os.path.join(config.paths.output_dir, prompt_type, "test.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Missing test CSV for variant {prompt_type}: {test_path}. "
                "Run 01_create_data.py first."
            )
        test_paths[prompt_type] = test_path

    # Lazy model load — only when needed for faithfulness.
    model = None

    for method in methods:
        print(f"\n=== Evaluating method: {method} ===")
        method_eval_dir = os.path.join(eval_root, method)
        os.makedirs(method_eval_dir, exist_ok=True)

        # Pre-check that all circuits needed across all aspects are present.
        needed = sorted({v for vs in aspects_cfg.values() for v in vs})
        circuit_paths_all: Dict[str, str] = {}
        missing = []
        for prompt_type in needed:
            path = os.path.join(circuits_root, method, f"{prompt_type}.pt")
            if not os.path.exists(path):
                missing.append(prompt_type)
            else:
                circuit_paths_all[prompt_type] = path
        if missing:
            print(
                f"  skipping method={method}: circuits missing for "
                f"variants {missing}. Run 02_run_circuit_discovery.py first."
            )
            continue

        for aspect, aspect_variants in aspects_cfg.items():
            print(f"\n  --- aspect: {aspect} ({len(aspect_variants)} variants) ---")
            aspect_dir = os.path.join(method_eval_dir, aspect)
            os.makedirs(aspect_dir, exist_ok=True)

            overlap_json = os.path.join(aspect_dir, "edge_overlap_matrix.json")
            faith_json = os.path.join(aspect_dir, "cross_faithfulness_matrix.json")

            circuit_paths = {v: circuit_paths_all[v] for v in aspect_variants}
            aspect_test_paths = {v: test_paths[v] for v in aspect_variants}

            # Edge overlap (fast, no model needed).
            if os.path.exists(overlap_json) and not args.force:
                print(f"    edge overlap already computed: {overlap_json}")
                with open(overlap_json) as f:
                    overlap = json.load(f)
            else:
                overlap = compute_edge_overlap_matrix(circuit_paths, GAMMA)
                with open(overlap_json, "w") as f:
                    json.dump(overlap, f, indent=2)
                print(f"    wrote {overlap_json}")

            # Cross-dataset faithfulness (requires model).
            if os.path.exists(faith_json) and not args.force:
                print(f"    faithfulness already computed: {faith_json}")
                with open(faith_json) as f:
                    faith = json.load(f)
            else:
                if model is None:
                    model, _ = load_model(config.model.name, config.model.cache_dir)
                    model.tokenizer.pad_token = model.tokenizer.eos_token
                faith = compute_cross_faithfulness_matrix(
                    model, circuit_paths, aspect_test_paths,
                    eval_batch_size=args.eval_batch_size,
                )
                with open(faith_json, "w") as f:
                    json.dump(faith, f, indent=2)
                print(f"    wrote {faith_json}")

            write_summary_tables(f"{method}/{aspect}", overlap, faith, aspect_dir)


if __name__ == "__main__":
    main()
