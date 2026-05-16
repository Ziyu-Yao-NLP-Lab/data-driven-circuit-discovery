"""Post-hoc purity vs k analysis for DCD clustering.

For the best combo of a given clustering method, sweeps every k from 2 to
max_clusters and computes:

  - Task-level purity     (majority task in each cluster, averaged across examples)
  - Prompt-type-level purity (majority prompt_type)
  - Silhouette score      (from results.json)

Generates a dual-axis figure:
  Left  y-axis — purity lines (task and prompt-type level)
  Right y-axis — silhouette score
  Vertical dashed lines mark the k chosen by silhouette, elbow, and gap
  selection criteria (whichever are available for this method).

Also prints a plain-text table to stdout.

IMPORTANT: This is post-hoc analysis only.  Purity is computed here to
understand the relationship between k and cluster quality.  It is never
used for k selection in the pipeline.

Usage:
    python experiments/dcd/analysis/purity_vs_k.py \\
        --config configs/dcd/all-tasks/llama-3.1-8b-instruct.yaml \\
        [--method hierarchical-agglomerative]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers — config / paths
# ---------------------------------------------------------------------------

def _results_root(config: OmegaConf) -> Path:
    task_name: str = config.data.task_name
    model_dir: str = Path(config.paths.output_dir).name
    return _ROOT / "results" / "dcd" / task_name / model_dir


def _is_all_tasks(config: OmegaConf) -> bool:
    return str(config.data.task_name) == "all-tasks"


def _build_prompt_to_task(config: OmegaConf) -> Dict[str, str]:
    """Build prompt_type → task_name mapping from config.data.tasks.

    Args:
        config: Loaded experiment config with a data.tasks section.

    Returns:
        Dict mapping prompt_type → task_name.  Empty if section absent.
    """
    mapping: Dict[str, str] = {}
    for task_name, task_cfg in config.data.get("tasks", {}).items():
        if task_cfg is None:
            continue
        for pt in task_cfg.get("prompt_types", []):
            mapping[str(pt)] = str(task_name)
    return mapping


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_train_labels(config: OmegaConf, results_root: Path) -> pd.DataFrame:
    """Load the training CSV, returning a DataFrame with a prompt_type column.

    Tries results/data/train.csv first (written by 02_run_attribution.py),
    then falls back to the original data location from the config.

    Args:
        config: Loaded experiment config.
        results_root: Root results directory for this task/model.

    Returns:
        DataFrame with prompt_type column, index reset.

    Raises:
        FileNotFoundError: If no train CSV is found.
        ValueError: If the prompt_type column is absent.
    """
    candidates = [
        results_root / "data" / "train.csv",
        results_root / "train_head.csv",
        Path(config.paths.output_dir) / "train.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path).reset_index(drop=True)
        if "prompt_type" not in df.columns:
            raise ValueError(f"prompt_type column not found in {path}")
        return df

    raise FileNotFoundError(
        "Could not find train CSV. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def _load_results_json(result_dir: Path) -> Dict[int, dict]:
    """Load results.json, returning a dict keyed by integer k.

    Args:
        result_dir: Clustering result directory containing results.json.

    Returns:
        Dict mapping k (int) → data dict (silhouette_score, inertia, …).

    Raises:
        FileNotFoundError: If results.json does not exist.
    """
    path = result_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"results.json not found at {path}")
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for k_str, v in raw.items():
        try:
            result[int(k_str.split("-")[0])] = v
        except (ValueError, IndexError):
            pass  # skip non-cluster keys like 'elbow_k'
    return result


# ---------------------------------------------------------------------------
# Purity computation
# ---------------------------------------------------------------------------

def _purity_for_groups(
    groups: np.ndarray,
    labels: pd.Series,
) -> float:
    """Compute overall purity for one grouping.

    Purity = fraction of examples in their cluster's majority class.

    Args:
        groups: Cluster assignment array, shape (n_examples,).  1-indexed.
        labels: Series of label strings aligned with groups.

    Returns:
        Purity score in [0, 1].
    """
    majority_correct = 0
    for cid in np.unique(groups):
        mask = groups == cid
        majority_correct += int(labels[mask].value_counts().iloc[0])
    return majority_correct / len(groups)


def _compute_purities(
    result_dir: Path,
    train_df: pd.DataFrame,
    prompt_to_task: Dict[str, str],
    k_values: List[int],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute task-level and prompt-type-level purity for each k.

    Args:
        result_dir: Directory containing groups_k{k}.npy files.
        train_df: Training DataFrame with prompt_type column.
        prompt_to_task: Mapping from prompt_type to task_name (may be empty
            for single-task configs, in which case task purity == pt purity).
        k_values: List of k values to evaluate.

    Returns:
        Tuple of (task_purities, pt_purities), each mapping k → purity float.
        Missing k values (no groups file or row mismatch) are silently skipped.
    """
    task_purities: Dict[int, float] = {}
    pt_purities: Dict[int, float] = {}
    n_train = len(train_df)
    pt_series = train_df["prompt_type"].reset_index(drop=True)

    if prompt_to_task:
        task_series = pt_series.map(lambda pt: prompt_to_task.get(pt, pt))
    else:
        task_series = pt_series  # single-task: task == prompt_type

    for k in k_values:
        groups_path = result_dir / f"groups_k{k}.npy"
        if not groups_path.exists():
            continue
        groups = np.load(groups_path)
        if len(groups) != n_train:
            # Edge scores may have been built on a subset; truncate labels
            if len(groups) > n_train:
                print(
                    f"  Warning: groups_k{k} has {len(groups)} rows > "
                    f"train ({n_train}). Skipping k={k}."
                )
                continue
            pt_sub = pt_series.iloc[: len(groups)].reset_index(drop=True)
            task_sub = task_series.iloc[: len(groups)].reset_index(drop=True)
        else:
            pt_sub = pt_series
            task_sub = task_series

        task_purities[k] = _purity_for_groups(groups, task_sub)
        pt_purities[k] = _purity_for_groups(groups, pt_sub)

    return task_purities, pt_purities


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Vertical-line style per selection criterion
_K_MARKERS = {
    "silhouette": dict(color="#0072B2", linestyle="--", linewidth=1.4,
                       label="k (silhouette)"),
    "elbow":      dict(color="#D55E00", linestyle="-.", linewidth=1.4,
                       label="k (elbow)"),
    "gap":        dict(color="#009E73", linestyle=":",  linewidth=1.6,
                       label="k (gap)"),
}


def _plot_purity_vs_k(
    k_values: List[int],
    task_purities: Dict[int, float],
    pt_purities: Dict[int, float],
    silhouettes: Dict[int, float],
    selected_k: Dict[str, int],
    method: str,
    task_name: str,
    model_dir: str,
    out_path: Path,
) -> None:
    """Generate and save the dual-axis purity vs k figure.

    Args:
        k_values: Sorted list of k values present in all three dicts.
        task_purities: k → task-level purity.
        pt_purities: k → prompt-type-level purity.
        silhouettes: k → silhouette score.
        selected_k: Dict of criterion → selected k (e.g. {"silhouette": 6}).
        method: Method name for subtitle.
        task_name: Config task_name for title.
        model_dir: Model directory name for title.
        out_path: Destination PNG path.
    """
    ks = sorted(set(k_values) & set(task_purities) & set(pt_purities))
    task_pur = [task_purities[k] for k in ks]
    pt_pur   = [pt_purities[k]   for k in ks]
    sil      = [silhouettes.get(k, np.nan) for k in ks]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # Purity lines (left axis)
    l1, = ax1.plot(ks, task_pur, color="#E69F00", marker="o", markersize=4,
                   linewidth=1.8, label="Task purity")
    l2, = ax1.plot(ks, pt_pur,   color="#CC79A7", marker="s", markersize=4,
                   linewidth=1.8, linestyle="--", label="Prompt-type purity")

    # Silhouette line (right axis)
    l3, = ax2.plot(ks, sil, color="#56B4E9", marker="^", markersize=4,
                   linewidth=1.6, linestyle="-.", label="Silhouette")

    # Vertical markers for selected k values
    vlines = []
    drawn_k: Dict[int, str] = {}  # avoid duplicate lines at same k
    for criterion, k_sel in selected_k.items():
        if k_sel not in drawn_k:
            style = _K_MARKERS[criterion]
            ax1.axvline(k_sel, **{kk: vv for kk, vv in style.items()
                                  if kk != "label"})
            drawn_k[k_sel] = criterion
        vlines.append(
            matplotlib.lines.Line2D([], [], **_K_MARKERS[criterion])
        )

    # Axes formatting
    ax1.set_xlabel("k (number of clusters)", fontsize=10)
    ax1.set_ylabel("Purity", fontsize=10, color="black")
    ax2.set_ylabel("Silhouette score", fontsize=10, color="#56B4E9")
    ax2.tick_params(axis="y", labelcolor="#56B4E9")

    ax1.set_xlim(min(ks) - 0.3, max(ks) + 0.3)
    ax1.set_xticks(ks)
    ax1.set_ylim(0, 1.05)
    ax1.spines["top"].set_visible(False)

    # Combined legend
    all_handles = [l1, l2, l3] + vlines
    ax1.legend(handles=all_handles, fontsize=8, loc="upper right",
               framealpha=0.85)

    ax1.set_title(
        f"{task_name} / {model_dir}  —  Purity vs k  ({method})",
        fontsize=11,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Stdout table
# ---------------------------------------------------------------------------

def _print_table(
    k_values: List[int],
    task_purities: Dict[int, float],
    pt_purities: Dict[int, float],
    silhouettes: Dict[int, float],
    selected_k: Dict[str, int],
) -> None:
    """Print a plain-text table of k, task_purity, pt_purity, silhouette.

    Args:
        k_values: Sorted list of k values.
        task_purities: k → task purity.
        pt_purities: k → prompt-type purity.
        silhouettes: k → silhouette score.
        selected_k: Criterion → selected k (for marking rows).
    """
    selected_ks = set(selected_k.values())
    header = f"{'k':>4}  {'task_purity':>12}  {'pt_purity':>10}  {'silhouette':>11}  note"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for k in sorted(k_values):
        tp  = task_purities.get(k, float("nan"))
        ptp = pt_purities.get(k, float("nan"))
        sil = silhouettes.get(k, float("nan"))
        marks = [crit for crit, kk in selected_k.items() if kk == k]
        note = ", ".join(f"<-- {m}" for m in marks) if marks else ""
        print(
            f"{k:>4}  {tp:>12.4f}  {ptp:>10.4f}  "
            f"{sil:>11.4f}  {note}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc purity vs k analysis for DCD clustering.",
    )
    parser.add_argument("--config", required=True,
                        help="Path to model-specific YAML config")
    parser.add_argument("--method", default="hierarchical-agglomerative",
                        help="Clustering method name (default: hierarchical-agglomerative)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    task_name: str = config.data.task_name
    model_dir: str = Path(config.paths.output_dir).name
    results_root = _results_root(config)

    # ---- best_configs ----
    best_configs_path = results_root / "clustering" / "best_configs.json"
    if not best_configs_path.exists():
        print(f"ERROR: best_configs.json not found at {best_configs_path}")
        print("Run 03_run_clustering.py --stage select first.")
        sys.exit(1)

    with open(best_configs_path) as f:
        best_configs = json.load(f)

    if args.method not in best_configs:
        print(f"ERROR: method '{args.method}' not in best_configs.json.")
        print(f"Available methods: {list(best_configs.keys())}")
        sys.exit(1)

    entry = best_configs[args.method]
    best_combo: str = entry["best_combo"]
    result_dir = _ROOT / Path(entry["result_dir"])
    print(f"Method:      {args.method}")
    print(f"Best combo:  {best_combo}")
    print(f"Result dir:  {result_dir}")

    # Collect selected-k markers (silhouette always present; elbow/gap optional)
    selected_k: Dict[str, int] = {"silhouette": int(entry["best_k"])}
    if "best_k_elbow" in entry:
        selected_k["elbow"] = int(entry["best_k_elbow"])
    if "best_k_gap" in entry:
        selected_k["gap"] = int(entry["best_k_gap"])
    print(f"Selected k:  {selected_k}")

    # ---- max_clusters from clustering config ----
    clustering_config_path = _ROOT / "configs" / "dcd" / "clustering_grid.yaml"
    max_clusters: int = 20  # safe default
    if clustering_config_path.exists():
        cc = OmegaConf.load(clustering_config_path)
        max_clusters = int(cc.clustering.max_clusters)
    k_values = list(range(2, max_clusters + 1))

    # ---- Load results.json for silhouette per k ----
    results_data = _load_results_json(result_dir)
    silhouettes: Dict[int, float] = {
        k: float(v.get("silhouette_score", float("nan")))
        for k, v in results_data.items()
        if 2 <= k <= max_clusters
    }

    # ---- Load train labels ----
    print("Loading train labels ...")
    train_df = _load_train_labels(config, results_root)
    prompt_to_task = _build_prompt_to_task(config) if _is_all_tasks(config) else {}
    print(f"  {len(train_df)} examples, "
          f"{len(train_df['prompt_type'].unique())} prompt types")

    # ---- Compute purities ----
    print(f"Computing purities for k=2..{max_clusters} ...")
    task_purities, pt_purities = _compute_purities(
        result_dir, train_df, prompt_to_task, k_values
    )
    print(f"  Computed for {len(task_purities)} k values.")

    # ---- Print table ----
    _print_table(k_values, task_purities, pt_purities, silhouettes, selected_k)

    # ---- Plot ----
    out_path = (results_root / "analysis"
                / f"purity_vs_k_{args.method}.png")
    _plot_purity_vs_k(
        k_values=k_values,
        task_purities=task_purities,
        pt_purities=pt_purities,
        silhouettes=silhouettes,
        selected_k=selected_k,
        method=args.method,
        task_name=task_name,
        model_dir=model_dir,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
