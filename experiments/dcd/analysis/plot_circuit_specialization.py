"""Plot per-circuit per-example specialization heatmaps for DCD vs K-Random.

Pure loader: reads the per-example faithfulness matrices saved by
``05_evaluate_faithfulness.py`` (DCD + K-Random), computes per-example CPR
and CMD via trapezoidal integration over circuit sizes, sorts test examples
by (task, prompt_type), and writes heatmaps + summary tables. Outputs:
    - circuit_specialization_cmd.png — trapezoidal CMD
    - circuit_specialization_cpr.png — trapezoidal CPR
    - circuit_specialization_faith_{size}.png — per-example faithfulness
      sliced at each circuit size strictly below 0.5 (one figure per size).

Usage:
    python experiments/dcd/analysis/plot_circuit_specialization.py \\
        --config configs/dcd/all-tasks/gpt2.yaml \\
        [--method kmeans-pca-raw]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Style — Wong colour-blind-safe palette, NeurIPS double-column sizing
# ---------------------------------------------------------------------------

_WONG = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "pink":       "#CC79A7",
    "sky":        "#56B4E9",
    "vermillion": "#D55E00",
    "yellow":     "#F0E442",
    "black":      "#000000",
}

# Cycle of strip colours for tasks (assigned in task order)
_STRIP_PALETTE = [
    _WONG["vermillion"], _WONG["sky"], _WONG["orange"],
    _WONG["green"],      _WONG["pink"], _WONG["blue"],
]

mpl.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _results_root(config: OmegaConf) -> Path:
    task_name: str = config.data.task_name
    model_dir: str = Path(config.paths.output_dir).name
    return _ROOT / "results" / "dcd" / task_name / model_dir


def _build_prompt_type_to_task(config: OmegaConf) -> Dict[str, str]:
    """Invert config.data.tasks: prompt_type -> task_name."""
    mapping: Dict[str, str] = {}
    for task_name, task_entry in config.data.tasks.items():
        for pt in task_entry.prompt_types:
            if pt in mapping and mapping[pt] != task_name:
                raise ValueError(
                    f"prompt_type '{pt}' appears under multiple tasks: "
                    f"{mapping[pt]} and {task_name}"
                )
            mapping[pt] = task_name
    return mapping


def _shorten_prompt_type(pt: str) -> str:
    """Shorten prompt_type names for display (e.g. '2-gram-symbolic' -> '2-gram')."""
    pt = pt.replace("-symbolic", "")
    return pt


# ---------------------------------------------------------------------------
# Sorting + labels
# ---------------------------------------------------------------------------

def _sort_test_examples(
    test_df: pd.DataFrame, pt_to_task: Dict[str, str]
) -> Tuple[np.ndarray, List[Tuple[str, int, int]], pd.DataFrame]:
    """Return sort order, (task, start, end) task_groups, and sorted DataFrame."""
    annotated = test_df.copy()
    annotated["task_name"] = annotated["prompt_type"].map(pt_to_task)
    missing = annotated.loc[annotated["task_name"].isna(), "prompt_type"].unique()
    if len(missing) > 0:
        raise ValueError(
            f"prompt_type(s) in test.csv not found in config.data.tasks: "
            f"{sorted(missing.tolist())}"
        )

    order = np.lexsort(
        (annotated["prompt_type"].values, annotated["task_name"].values)
    )
    sorted_df = annotated.iloc[order].reset_index(drop=True)

    task_groups: List[Tuple[str, int, int]] = []
    for task_name, grp in sorted_df.groupby("task_name", sort=False):
        task_groups.append(
            (str(task_name), int(grp.index.min()), int(grp.index.max()) + 1)
        )
    return order, task_groups, sorted_df


_CLUSTER_RE = re.compile(r"cluster_(\d+)")
_SPLIT_RE = re.compile(r"split_(\d+)")


def _parse_circuit_ids(filenames: np.ndarray, pattern: re.Pattern) -> List[int]:
    ids = []
    for name in filenames:
        m = pattern.search(str(name))
        if not m:
            raise ValueError(f"Cannot parse circuit id from filename: {name}")
        ids.append(int(m.group(1)))
    return ids


def _load_dcd_cluster_labels(
    results_root: Path, method: str, k: int
) -> Dict[int, str]:
    """Map DCD cluster_id -> majority prompt_type.

    First tries cluster_purity_full.csv; falls back to reading the cluster
    CSV files directly from circuits/created/dcd/{method}/.
    """
    path = results_root / "analysis" / "cluster_purity_full.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[(df["method"] == method) & (df["k"] == k)]
        if not df.empty:
            first_combo = sorted(df["combo"].unique())[0]
            df = df[df["combo"] == first_combo]
            return {int(r["cluster_id"]): str(r["majority_type"]) for _, r in df.iterrows()}

    # Fallback: derive from cluster CSV files
    cluster_dir = results_root / "circuits" / "created" / "dcd" / method
    if not cluster_dir.exists():
        return {}
    labels: Dict[int, str] = {}
    for csv_path in sorted(cluster_dir.glob("cluster_*.csv")):
        m = re.search(r"cluster_(\d+)", csv_path.stem)
        if not m:
            continue
        cid = int(m.group(1))
        df = pd.read_csv(csv_path)
        if "prompt_type" not in df.columns or df.empty:
            continue
        majority = df["prompt_type"].value_counts().idxmax()
        labels[cid] = str(majority)
    return labels


# ---------------------------------------------------------------------------
# Per-example metric computation
# ---------------------------------------------------------------------------

def _per_example_cpr(faith: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Trapezoidal CPR for each (circuit, example).

    Args:
        faith: shape (k, n_sizes, n_test).
        sizes: shape (n_sizes,).

    Returns:
        cpr: shape (k, n_test).
    """
    k, n_sizes, n_test = faith.shape
    cpr = np.zeros((k, n_test), dtype=np.float32)
    for i in range(n_sizes - 1):
        x1 = float(sizes[i])
        x2 = float(sizes[i + 1])
        cpr += (x2 - x1) * (faith[:, i, :] + faith[:, i + 1, :]) / 2.0
    return cpr


def _per_example_cmd(faith: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Trapezoidal CMD (one-sided) for each (circuit, example).

    Args:
        faith: shape (k, n_sizes, n_test).
        sizes: shape (n_sizes,).

    Returns:
        cmd: shape (k, n_test).
    """
    k, n_sizes, n_test = faith.shape
    dev = np.maximum(1.0 - faith, 0.0)
    cmd = np.zeros((k, n_test), dtype=np.float32)
    for i in range(n_sizes - 1):
        x1 = float(sizes[i])
        x2 = float(sizes[i + 1])
        cmd += (x2 - x1) * (dev[:, i, :] + dev[:, i + 1, :]) / 2.0
    return cmd


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_heatmaps(
    dcd_matrix: np.ndarray,
    random_matrix: np.ndarray,
    metric: str,
    task_groups: List[Tuple[str, int, int]],
    pt_groups: List[Tuple[str, str, int, int]],
    task_strip: np.ndarray,
    strip_colors: List[str],
    dcd_y_labels: List[str],
    random_y_labels: List[str],
    out_path: Path,
    size: float = None,
) -> None:
    """Side-by-side heatmaps with task colour strips and prompt-type labels.

    Args:
        dcd_matrix: (k, n_test) metric values for DCD circuits.
        random_matrix: (k, n_test) metric values for K-Random circuits.
        metric: "cmd", "cpr", or "faith" — controls colormap, clipping, and label.
        task_groups: List of (task_name, start, end) in sorted order.
        pt_groups: List of (task_name, pt_display, start, end) in sorted order.
        task_strip: (1, n_test) int array mapping each example to a task index.
        strip_colors: One hex color per task (same order as task_groups).
        dcd_y_labels: Y-axis labels for DCD circuits.
        random_y_labels: Y-axis labels for K-Random circuits.
        out_path: Where to save the figure.
        size: Circuit size (fraction of edges) — required when metric == "faith"
            so the colorbar can label which size the faithfulness is taken at.
    """
    strip_cmap = ListedColormap(strip_colors)

    # Metric-specific settings
    if metric == "cmd":
        cmap = "YlGnBu"
        vmin, vmax = 0.0, 0.5
        cbar_label = "Per-example CMD"
        extend = "max"
    elif metric == "cpr":
        cmap = "inferno"
        vmin, vmax = -0.3, 1.2
        cbar_label = "Per-example CPR"
        extend = "both"
    elif metric == "faith":
        if size is None:
            raise ValueError("metric='faith' requires the `size` argument.")
        cmap = "YlGnBu"
        vmin, vmax = 0.0, 1.0
        cbar_label = f"Per-example faithfulness (size={size:g})"
        extend = "both"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    fig = plt.figure(figsize=(7.0, 3.2))

    left_l,  left_r  = 0.08, 0.435
    right_l, right_r = 0.475, 0.83
    strip_h     = 0.03
    heat_bottom = 0.08
    heat_h      = 0.72
    strip_bottom = heat_bottom + heat_h + 0.005

    ax_sl = fig.add_axes([left_l,  strip_bottom, left_r  - left_l,  strip_h])
    ax_sr = fig.add_axes([right_l, strip_bottom, right_r - right_l, strip_h])
    ax_dl = fig.add_axes([left_l,  heat_bottom,  left_r  - left_l,  heat_h])
    ax_dr = fig.add_axes([right_l, heat_bottom,  right_r - right_l, heat_h])
    ax_cb = fig.add_axes([0.86,    heat_bottom,  0.015,             heat_h])

    n_tasks = len(task_groups)

    # -- Task colour strips --
    for ax_s in [ax_sl, ax_sr]:
        ax_s.imshow(
            task_strip, aspect="auto", cmap=strip_cmap,
            vmin=-0.5, vmax=n_tasks - 0.5, interpolation="nearest",
        )
        ax_s.set_yticks([])
        ax_s.set_xticks([])
        for sp in ax_s.spines.values():
            sp.set_visible(False)

        # Prompt-type labels (shortened)
        for _, pt, s, e in pt_groups:
            ax_s.text(
                (s + e - 1) / 2, -0.9, _shorten_prompt_type(pt),
                ha="center", va="bottom", fontsize=5.5, color="#333333",
            )
        # Task name labels
        for task, s, e in task_groups:
            ax_s.text(
                (s + e - 1) / 2, -2.8, task,
                ha="center", va="bottom", fontsize=7,
                fontweight="bold", color="black",
            )

    # -- Heatmaps --
    for ax, matrix, y_labels, xlabel in [
        (ax_dl, dcd_matrix,    dcd_y_labels,    "DCD circuits"),
        (ax_dr, random_matrix, random_y_labels, "Random $k$-split"),
    ]:
        im = ax.imshow(
            matrix, aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(y_labels)
        ax.set_xticks([])

        # Prompt-type dividers (thin dashed)
        for _, _, _, end in pt_groups[:-1]:
            ax.axvline(end - 0.5, color="white", lw=0.4, ls="--", alpha=0.5)
        # Task dividers (solid)
        for _, _, end in task_groups[:-1]:
            ax.axvline(end - 0.5, color="white", lw=1.0)

        ax.set_xlabel(xlabel, fontsize=8, fontweight="bold")

    ax_dl.set_ylabel("Circuit")
    ax_dr.set_yticklabels([])

    cb = fig.colorbar(im, cax=ax_cb, extend=extend)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _write_summary(
    dcd_matrix: np.ndarray,
    metric: str,
    cluster_ids: List[int],
    cluster_labels: Dict[int, str],
    test_sorted_prompt_types: np.ndarray,
    out_path: Path,
) -> None:
    """Per DCD circuit: own-cluster metric mean, other metric mean, gap."""
    header = (
        f"{'circuit_id':<24} {f'mean_own_{metric}':>22} "
        f"{f'mean_other_{metric}':>18} {'specialization_gap':>20}"
    )
    lines = [header, "-" * len(header)]

    for row, cid in enumerate(cluster_ids):
        label = cluster_labels.get(cid)
        row_scores = dcd_matrix[row]
        if label is None:
            own_mean = float("nan")
            other_mean = float(row_scores.mean())
            gap = float("nan")
            cid_str = f"C{cid}"
        else:
            own_mask = test_sorted_prompt_types == label
            if own_mask.sum() == 0:
                own_mean = float("nan")
                other_mean = float(row_scores.mean())
                gap = float("nan")
            else:
                own_mean = float(row_scores[own_mask].mean())
                other_mean = float(row_scores[~own_mask].mean())
                gap = own_mean - other_mean
            cid_str = f"C{cid}: {label}"
        lines.append(
            f"{cid_str:<24} {own_mean:>22.4f} {other_mean:>18.4f} {gap:>20.4f}"
        )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DCD vs K-Random per-circuit specialization heatmaps (CPR + CMD).",
    )
    parser.add_argument("--config", required=True, help="Path to DCD model config YAML.")
    parser.add_argument(
        "--method",
        default="kmeans-pca-raw",
        help="DCD clustering method whose npz to load.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    results_root = _results_root(config)
    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = analysis_dir / "evaluation"
    dcd_npz = eval_dir / "dcd" / f"{args.method}.npz"
    random_npz = eval_dir / "random_k_split.npz"

    if not dcd_npz.exists():
        raise FileNotFoundError(
            f"DCD per-example matrix not found: {dcd_npz}\n"
            "Run experiments/dcd/05_evaluate_faithfulness.py --circuit-type dcd "
            "first (it now writes per-example matrices alongside the aggregated "
            "evaluation results)."
        )
    if not random_npz.exists():
        raise FileNotFoundError(
            f"K-Random per-example matrix not found: {random_npz}\n"
            "Run experiments/dcd/05_evaluate_faithfulness.py --circuit-type random_k_split first."
        )

    dcd_data    = np.load(dcd_npz,    allow_pickle=False)
    random_data = np.load(random_npz, allow_pickle=False)

    dcd_faith    = dcd_data["faith"]        # (k_dcd,  n_sizes, n_test)
    random_faith = random_data["faith"]     # (k_rand, n_sizes, n_test)
    dcd_sizes    = dcd_data["sizes"]
    random_sizes = random_data["sizes"]

    if not np.array_equal(dcd_sizes, random_sizes):
        raise ValueError(
            f"DCD and K-Random size grids differ:\n  dcd={dcd_sizes}\n  random={random_sizes}"
        )

    dcd_files    = dcd_data["circuit_files"]
    random_files = random_data["circuit_files"]
    dcd_ids    = _parse_circuit_ids(dcd_files,    _CLUSTER_RE)
    random_ids = _parse_circuit_ids(random_files, _SPLIT_RE)

    print(
        f"DCD: {dcd_faith.shape}  K-Random: {random_faith.shape}  "
        f"sizes={list(dcd_sizes)}"
    )

    # Compute both metrics
    dcd_cmd    = _per_example_cmd(dcd_faith,    dcd_sizes)
    random_cmd = _per_example_cmd(random_faith, random_sizes)
    dcd_cpr    = _per_example_cpr(dcd_faith,    dcd_sizes)
    random_cpr = _per_example_cpr(random_faith, random_sizes)

    # Test data + sort
    test_csv = Path(config.paths.output_dir) / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found at {test_csv}")
    test_df = pd.read_csv(test_csv)

    n_test = len(test_df)
    for name, mat in [("DCD CMD", dcd_cmd), ("Random CMD", random_cmd),
                       ("DCD CPR", dcd_cpr), ("Random CPR", random_cpr)]:
        if mat.shape[1] != n_test:
            raise RuntimeError(
                f"{name} column count ({mat.shape[1]}) does not match "
                f"test.csv rows ({n_test}). Re-run 05_evaluate_faithfulness.py --force."
            )

    pt_to_task = _build_prompt_type_to_task(config)
    sort_idx, task_groups, test_sorted = _sort_test_examples(test_df, pt_to_task)
    print(f"Test examples: {n_test}  tasks: {[t for t, _, _ in task_groups]}")

    dcd_cmd_sorted    = dcd_cmd[:,    sort_idx]
    random_cmd_sorted = random_cmd[:, sort_idx]
    dcd_cpr_sorted    = dcd_cpr[:,    sort_idx]
    random_cpr_sorted = random_cpr[:, sort_idx]

    # Per-prompt-type groups for strip labels
    pt_groups: List[Tuple[str, str, int, int]] = []
    for (task, pt), grp in test_sorted.groupby(
        ["task_name", "prompt_type"], sort=False
    ):
        pt_groups.append(
            (str(task), str(pt), int(grp.index.min()), int(grp.index.max()) + 1)
        )

    # Task colour strip — integer index per example
    task_order = {task: i for i, (task, _, _) in enumerate(task_groups)}
    task_strip = np.array(
        [task_order[t] for t in test_sorted["task_name"]]
    )[np.newaxis, :]
    strip_colors = [_STRIP_PALETTE[i % len(_STRIP_PALETTE)] for i in range(len(task_groups))]

    # Y-axis labels
    cluster_labels = _load_dcd_cluster_labels(results_root, args.method, len(dcd_ids))
    dcd_y_labels = [f"$c_{{{i+1}}}$" for i in range(len(dcd_ids))]
    random_y_labels = [f"$c_{{{i+1}}}$" for i in range(len(random_ids))]

    # Shared args for both plots
    shared_kwargs = dict(
        task_groups=task_groups,
        pt_groups=pt_groups,
        task_strip=task_strip,
        strip_colors=strip_colors,
        dcd_y_labels=dcd_y_labels,
        random_y_labels=random_y_labels,
    )

    # ── CMD figure ─────────────────────────────────────────────────────
    cmd_plot_path = analysis_dir / "circuit_specialization_cmd.png"
    print(f"Writing {cmd_plot_path}")
    _plot_heatmaps(
        dcd_matrix=dcd_cmd_sorted,
        random_matrix=random_cmd_sorted,
        metric="cmd",
        out_path=cmd_plot_path,
        **shared_kwargs,
    )

    cmd_summary_path = analysis_dir / "circuit_specialization_cmd_summary.txt"
    print(f"Writing {cmd_summary_path}")
    _write_summary(
        dcd_cmd_sorted, "cmd", dcd_ids, cluster_labels,
        test_sorted["prompt_type"].to_numpy(), cmd_summary_path,
    )

    # ── CPR figure ─────────────────────────────────────────────────────
    cpr_plot_path = analysis_dir / "circuit_specialization_cpr.png"
    print(f"Writing {cpr_plot_path}")
    _plot_heatmaps(
        dcd_matrix=dcd_cpr_sorted,
        random_matrix=random_cpr_sorted,
        metric="cpr",
        out_path=cpr_plot_path,
        **shared_kwargs,
    )

    cpr_summary_path = analysis_dir / "circuit_specialization_cpr_summary.txt"
    print(f"Writing {cpr_summary_path}")
    _write_summary(
        dcd_cpr_sorted, "cpr", dcd_ids, cluster_labels,
        test_sorted["prompt_type"].to_numpy(), cpr_summary_path,
    )

    # ── Faithfulness-at-size figures (one per size < 0.5) ──────────────
    sizes_below = [(i, float(s)) for i, s in enumerate(dcd_sizes) if float(s) < 0.5]
    print(
        f"Generating faithfulness-at-size figures for {len(sizes_below)} "
        f"sizes < 0.5: {[f'{s:g}' for _, s in sizes_below]}"
    )
    for size_idx, size_val in sizes_below:
        dcd_at    = dcd_faith[:,    size_idx, :][:, sort_idx]
        random_at = random_faith[:, size_idx, :][:, sort_idx]

        size_str = f"{size_val:g}"
        plot_path = analysis_dir / f"circuit_specialization_faith_{size_str}.png"
        print(f"Writing {plot_path}")
        _plot_heatmaps(
            dcd_matrix=dcd_at,
            random_matrix=random_at,
            metric="faith",
            size=size_val,
            out_path=plot_path,
            **shared_kwargs,
        )

        summary_path = analysis_dir / f"circuit_specialization_faith_{size_str}_summary.txt"
        print(f"Writing {summary_path}")
        _write_summary(
            dcd_at, f"faith_{size_str}", dcd_ids, cluster_labels,
            test_sorted["prompt_type"].to_numpy(), summary_path,
        )

    print("Done.")


if __name__ == "__main__":
    main()