"""Post-hoc cluster purity analysis for the DCD pipeline.

Walks the clustering results directory tree, loads cluster assignments and
train prompt_type labels, and computes per-cluster and overall purity scores.

This script is analysis-only: it reads outputs produced by 03_run_clustering.py
but never feeds back into the pipeline.  Prompt type labels are never used for
hyperparameter selection.

Usage:
    python experiments/dcd/analysis/cluster_purity.py \\
        --config configs/dcd/all-tasks/llama-3.1-8b-instruct.yaml \\
        [--method hierarchical-agglomerative]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_train_data(config: OmegaConf, results_root: Path) -> pd.DataFrame:
    """Load the training CSV that was used during clustering.

    Tries train_head.csv (written by 02_run_attribution.py) first, then falls
    back to the full train.csv from the data output directory.

    Args:
        config: Loaded experiment config.
        results_root: Root results directory for this task/model.

    Returns:
        DataFrame with at least a ``prompt_type`` column, index reset.

    Raises:
        FileNotFoundError: If neither candidate CSV exists.
        ValueError: If prompt_type column is absent from the loaded CSV.
    """
    candidates = [
        results_root / "train_head.csv",
        Path(config.paths.output_dir) / "train.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path).reset_index(drop=True)
            if "prompt_type" not in df.columns:
                raise ValueError(
                    f"prompt_type column not found in {path}. "
                    f"Columns present: {list(df.columns)}"
                )
            return df

    raise FileNotFoundError(
        "Could not find train data. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def _find_results_jsons(
    clustering_root: Path,
    method_filter: Optional[str],
) -> List[Tuple[str, str, str, Path]]:
    """Walk the clustering directory and return all results.json paths.

    Expected layout:
        clustering_root/{method}/{combo}/{result_key}/results.json

    Args:
        clustering_root: Root of the clustering output tree.
        method_filter: If set, only include entries whose method equals this.

    Returns:
        List of (method, combo, result_key, path) tuples.
    """
    found = []
    for json_path in sorted(clustering_root.rglob("results.json")):
        rel = json_path.relative_to(clustering_root)
        parts = rel.parts
        if len(parts) != 4:  # method / combo / result_key / results.json
            continue
        method, combo, result_key = parts[0], parts[1], parts[2]
        if method_filter is not None and method != method_filter:
            continue
        found.append((method, combo, result_key, json_path))
    return found


# ---------------------------------------------------------------------------
# Purity computation
# ---------------------------------------------------------------------------

def _compute_purity(
    groups: np.ndarray,
    prompt_types: pd.Series,
    all_prompt_types: List[str],
) -> Tuple[List[dict], float]:
    """Compute per-cluster purity and prompt-type distributions.

    Args:
        groups: Cluster assignment array, shape (n_samples,).
        prompt_types: Series of prompt_type labels aligned with groups.
        all_prompt_types: Sorted list of all unique prompt types (column order).

    Returns:
        Tuple of:
        - per_cluster: list of dicts, one per cluster, with keys
          cluster_id, cluster_size, majority_type, purity, and one key
          per prompt type holding the percentage (0–100).
        - overall_purity: fraction of examples in their cluster's majority
          class.
    """
    per_cluster = []
    majority_correct = 0

    for cid in np.unique(groups):
        mask = groups == cid
        cluster_size = int(mask.sum())
        cluster_types = prompt_types[mask]

        counts = cluster_types.value_counts()
        majority_type = counts.index[0]
        majority_count = int(counts.iloc[0])
        cluster_purity = majority_count / cluster_size
        majority_correct += majority_count

        row: dict = {
            "cluster_id": int(cid),
            "cluster_size": cluster_size,
            "majority_type": majority_type,
            "purity": cluster_purity,
        }
        for pt in all_prompt_types:
            row[pt] = 100.0 * counts.get(pt, 0) / cluster_size

        per_cluster.append(row)

    overall_purity = majority_correct / len(groups)
    return per_cluster, overall_purity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc cluster purity analysis for DCD clustering results.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model-specific YAML config (same as used for 03_run_clustering.py)",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="If set, restrict analysis to this clustering method name.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    task_name: str = config.data.task_name
    model_dir: str = Path(config.paths.output_dir).name
    results_root = _ROOT / "results" / "dcd" / task_name / model_dir
    clustering_root = results_root / "clustering"

    if not clustering_root.exists():
        print(
            f"Clustering directory not found: {clustering_root}\n"
            "Run 03_run_clustering.py --stage cluster first."
        )
        sys.exit(1)

    # Load train data
    train_data = _load_train_data(config, results_root)
    prompt_types_series = train_data["prompt_type"].reset_index(drop=True)
    all_prompt_types = sorted(prompt_types_series.unique().tolist())
    print(f"Loaded {len(train_data)} training examples with prompt types: {all_prompt_types}")

    # Discover all results.json files
    entries = _find_results_jsons(clustering_root, args.method)
    if not entries:
        print(
            f"No results.json files found under {clustering_root}"
            + (f" for method '{args.method}'" if args.method else "")
        )
        sys.exit(1)

    print(f"Found {len(entries)} result entries to analyse.")

    full_rows: List[dict] = []
    summary_rows: List[dict] = []

    for method, combo, result_key, json_path in entries:
        result_dir = json_path.parent

        with open(json_path) as f:
            results_data: dict = json.load(f)

        for k_key, k_data in results_data.items():
            # k_key format: "2-clusters"
            try:
                k = int(k_key.split("-")[0])
            except (ValueError, IndexError):
                print(
                    f"ERROR: unexpected key '{k_key}' in {json_path}.\n"
                    "results.json appears corrupted. Aborting."
                )
                sys.exit(1)

            groups_path = result_dir / f"groups_k{k}.npy"
            if not groups_path.exists():
                print(
                    f"ERROR: {groups_path} not found but k={k} is listed in "
                    f"{json_path}.\nResults appear incomplete. Aborting."
                )
                sys.exit(1)

            groups = np.load(groups_path)

            if len(groups) != len(prompt_types_series):
                print(
                    f"ERROR: groups_k{k}.npy has {len(groups)} rows but train data "
                    f"has {len(prompt_types_series)} rows ({method}/{combo}).\n"
                    "Cluster assignments and train data are mismatched. Aborting."
                )
                sys.exit(1)

            silhouette = k_data.get("silhouette_score", float("nan"))
            per_cluster, overall_purity = _compute_purity(
                groups, prompt_types_series, all_prompt_types
            )

            for cluster_row in per_cluster:
                full_rows.append({
                    "method": method,
                    "combo": combo,
                    "k": k,
                    **cluster_row,
                })

            summary_rows.append({
                "method": method,
                "combo": combo,
                "k": k,
                "overall_purity": overall_purity,
                "silhouette": silhouette,
            })

    if not summary_rows:
        print("No valid results found. Nothing to save.")
        sys.exit(1)

    out_dir = results_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "overall_purity", ascending=False
    ).reset_index(drop=True)

    full_path = out_dir / "cluster_purity_full.csv"
    summary_path = out_dir / "cluster_purity_summary.csv"
    full_df.to_csv(full_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {full_path}  ({len(full_df)} rows)")
    print(f"Saved {summary_path}  ({len(summary_df)} rows)")

    # Print top-10 summary table
    top10 = summary_df.head(10)
    col_w = {"method": 35, "combo": 30, "k": 4, "overall_purity": 13, "silhouette": 10}
    header = (
        f"{'method':<{col_w['method']}}  "
        f"{'combo':<{col_w['combo']}}  "
        f"{'k':>{col_w['k']}}  "
        f"{'overall_purity':>{col_w['overall_purity']}}  "
        f"{'silhouette':>{col_w['silhouette']}}"
    )
    sep = "-" * len(header)
    print(f"\nTop 10 method/combo/k by purity:\n{sep}\n{header}\n{sep}")
    for _, row in top10.iterrows():
        sil = row["silhouette"]
        sil_str = f"{sil:.4f}" if not (isinstance(sil, float) and np.isnan(sil)) else "   nan"
        print(
            f"{row['method']:<{col_w['method']}}  "
            f"{row['combo']:<{col_w['combo']}}  "
            f"{int(row['k']):>{col_w['k']}}  "
            f"{row['overall_purity']:>{col_w['overall_purity']}.4f}  "
            f"{sil_str:>{col_w['silhouette']}}"
        )
    print(sep)


if __name__ == "__main__":
    main()
