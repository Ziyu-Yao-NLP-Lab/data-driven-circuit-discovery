"""Three-panel figure: cluster composition + DCD/Random circuit specialization.

Combines the bar chart from ``plot_cluster_composition.py`` with the two heatmaps
from ``plot_circuit_specialization.py`` into a single figure with a shared
cluster axis (c1...c_k running top→bottom).

Layout:
    [Panel A: horizontal stacked bars] | [Panel B: DCD heatmap] | [Panel C: Random heatmap] | [colorbar]
    (cluster composition)              | (per-example faith)    | (per-example faith, baseline)
                              [shared prompt-type legend below]

Usage:
    python experiments/dcd/analysis/plot_specialization_3panel.py \
        --config configs/dcd/all-tasks/gpt2.yaml \
        [--method kmeans-pca] [--size 0.05]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Style ────────────────────────────────────────────────────
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

_STRIP_PALETTE = [
    _WONG["vermillion"], _WONG["sky"], _WONG["orange"],
    _WONG["green"],      _WONG["pink"], _WONG["blue"],
]

# Per-task palettes for the bar chart (matches plot_cluster_composition.py)
_TASK_PALETTES = [
    ["#8b2323", "#c44e52", "#e8927c", "#f0c0a0", "#f5dcc8", "#fae8dc"],  # warm reds
    ["#1a5276", "#2980b9", "#4c9bd6", "#7ec4e8", "#b0ddf0", "#d4eef8"],  # cool blues
    ["#1e6e3e", "#27ae60", "#52c77e", "#82d9a0", "#b2e8c4", "#d8f4e0"],
    ["#4a235a", "#7d3c98", "#a569bd", "#c39bd3", "#d7bde2", "#ebdef0"],
    ["#7e3f00", "#d35400", "#e67e22", "#f0a04b", "#f5c27a", "#fae0b0"],
    ["#0e4d4d", "#138d8d", "#1abc9c", "#48d1b5", "#82e0cd", "#bef0e5"],
]

mpl.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ── Helpers (mirrored from sibling scripts) ───────────────────
def _strip_image_ext(s: str) -> str:
    return s[:-4] if s.endswith(".png") or s.endswith(".pdf") else s


def _pdf_sibling(png_path: Path) -> Path:
    # Path.with_suffix breaks on dotted stems (e.g. "llama-3.1-8b-instruct").
    name = png_path.name
    base = name[:-4] if name.endswith(".png") else name
    return png_path.parent / f"{base}.pdf"


def _results_root(config: OmegaConf) -> Path:
    return _ROOT / "results" / "dcd" / config.data.task_name / Path(config.paths.output_dir).name


def _build_prompt_type_to_task(config: OmegaConf) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for task_name, task_entry in config.data.tasks.items():
        for pt in task_entry.prompt_types:
            mapping[pt] = task_name
    return mapping


_PROMPT_DISPLAY_OVERRIDES = {
    "mixed": "2-person",
}


def _shorten_prompt(pt: str) -> str:
    if pt in _PROMPT_DISPLAY_OVERRIDES:
        return _PROMPT_DISPLAY_OVERRIDES[pt]
    return pt.replace("-symbolic", "")


# Aggressive abbreviations for strip headers (column groups are narrow and
# long names overlap their neighbors). Full readable names stay in the legend.
_STRIP_ALIASES = {
    "2-operand-addition":            "2-op-add",
    "3-operand-addition":            "3-op-add",
    "2-operand-addition-verbal-v1":  "verbal",
    "2-operand-addition-verbal-v2":  "verbal-v2",
    "2-operand-addition-verbal-v3":  "verbal-v3",
    "2-color-box-comma":             "color-box",
    "2-comma":                       "2-comma",
    "comma-0-position":              "pos-0",
    "comma-7-position":              "pos-7",
}


def _strip_prompt(pt: str) -> str:
    return _STRIP_ALIASES.get(pt, _shorten_prompt(pt))


def _shorten_task(name: str) -> str:
    return {"sequence-completion": "SC", "ioi": "IOI",
            "arithmetic": "Arith", "entity-binding": "EB"}.get(name, name.upper()[:6])


def _generate_prompt_colors(task_prompt_map: Dict[str, List[str]]):
    """Return (prompt_to_task, prompt_colors, prompt_order)."""
    prompt_to_task, prompt_colors, prompt_order = {}, {}, []
    for ti, (task, pts) in enumerate(task_prompt_map.items()):
        palette = _TASK_PALETTES[ti % len(_TASK_PALETTES)]
        for pi, pt in enumerate(pts):
            prompt_to_task[pt] = task
            prompt_colors[pt] = palette[min(pi + 1, len(palette) - 1)]
            prompt_order.append(pt)
    return prompt_to_task, prompt_colors, prompt_order


def _load_cluster_csvs(cluster_dir: Path) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for f in sorted(cluster_dir.glob("cluster_*.csv")):
        m = re.search(r"cluster_(\d+)", f.stem)
        if m:
            out[int(m.group(1))] = pd.read_csv(f)
    if not out:
        raise FileNotFoundError(f"No cluster_*.csv files in {cluster_dir}")
    return out


def _sort_test(test_df: pd.DataFrame, pt_to_task: Dict[str, str]):
    annotated = test_df.copy()
    annotated["task_name"] = annotated["prompt_type"].map(pt_to_task)
    if annotated["task_name"].isna().any():
        bad = annotated.loc[annotated["task_name"].isna(), "prompt_type"].unique()
        raise ValueError(f"prompt_type(s) not in config: {sorted(bad.tolist())}")
    order = np.lexsort((annotated["prompt_type"].values, annotated["task_name"].values))
    sorted_df = annotated.iloc[order].reset_index(drop=True)
    task_groups = [(str(t), int(g.index.min()), int(g.index.max()) + 1)
                   for t, g in sorted_df.groupby("task_name", sort=False)]
    return order, task_groups, sorted_df


# ── Drawing primitives ──────────────────────────────────────
def _draw_bars(ax, cluster_data, cluster_ids, prompt_order, prompt_colors, prompt_to_task):
    """Horizontal stacked bars: prompt-type composition per cluster, c_1 at top."""
    k = len(cluster_ids)
    y_pos = np.arange(k)
    lefts = np.zeros(k)
    for pt in prompt_order:
        vals = np.array([
            cluster_data[cid]["prompt_type"].value_counts().get(pt, 0)
            for cid in cluster_ids
        ])
        ax.barh(
            y_pos, vals, left=lefts, color=prompt_colors[pt],
            edgecolor="white", linewidth=0.4, height=0.78,
            label=f"{prompt_to_task[pt]} / {_shorten_prompt(pt)}",
        )
        lefts += vals

    task_label_color = {}
    for ti, task in enumerate(dict.fromkeys(prompt_to_task.values())):
        task_label_color[task] = (
            _WONG["vermillion"] if ti == 0
            else _WONG["blue"] if ti == 1
            else _STRIP_PALETTE[ti % len(_STRIP_PALETTE)]
        )
    for j, cid in enumerate(cluster_ids):
        df = cluster_data[cid]
        task_counts: Dict[str, int] = {}
        for pt, n in df["prompt_type"].value_counts().items():
            t = prompt_to_task.get(pt, "unknown")
            task_counts[t] = task_counts.get(t, 0) + int(n)
        if not task_counts:
            continue
        dom = max(task_counts, key=task_counts.get)
        purity = task_counts[dom] / sum(task_counts.values())
        if purity >= 0.95:
            ax.text(
                lefts[j] + 2, j, _shorten_task(dom),
                ha="left", va="center", fontsize=9,
                fontweight="bold", color=task_label_color.get(dom, "black"),
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"$c_{{{cid}}}$" for cid in cluster_ids])
    ax.invert_yaxis()
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_xlabel("Number of examples")
    ax.set_ylabel("Circuit")
    ax.tick_params(axis="x", which="both", length=3)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def _draw_strip(ax_strip, task_strip, strip_cmap, n_tasks, pt_groups, task_groups):
    """Color strip + prompt-type and task labels above a heatmap."""
    ax_strip.imshow(
        task_strip, aspect="auto", cmap=strip_cmap,
        vmin=-0.5, vmax=n_tasks - 0.5, interpolation="nearest",
    )
    ax_strip.set_xticks([]); ax_strip.set_yticks([])
    for sp in ax_strip.spines.values():
        sp.set_visible(False)
    for _, pt, s, e in pt_groups:
        ax_strip.text(
            (s + e - 1) / 2, -1.0, _strip_prompt(pt),
            ha="center", va="bottom", fontsize=10, color="#333333",
        )
    for task, s, e in task_groups:
        ax_strip.text(
            (s + e - 1) / 2, -3.4, task,
            ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="black",
        )


def _draw_heatmap(ax, mat, cmap, vmin, vmax, pt_groups, task_groups):
    """Heatmap with prompt-type and task vertical separators. Returns the AxesImage."""
    im = ax.imshow(
        mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([])
    ax.set_xticks([])
    for _, _, _, end in pt_groups[:-1]:
        ax.axvline(end - 0.5, color="white", lw=0.4, ls="--", alpha=0.5)
    for _, _, end in task_groups[:-1]:
        ax.axvline(end - 0.5, color="white", lw=1.0)
    return im


def _draw_legend(fig, prompt_order, prompt_colors, prompt_to_task, ncol_max=6):
    """Shared prompt-type legend at the bottom of the figure."""
    handles = [
        mpatches.Patch(
            facecolor=prompt_colors[pt], edgecolor="white",
            label=f"{prompt_to_task[pt]} / {_shorten_prompt(pt)}",
        )
        for pt in prompt_order
    ]
    fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), ncol_max),
        frameon=False, fontsize=10,
        title="task / prompt type", title_fontsize=10,
        handlelength=1.6, columnspacing=1.6,
    )


# ── Plot ─────────────────────────────────────────────────────
def plot_three_panel(
    cluster_data: Dict[int, pd.DataFrame],
    dcd_at: np.ndarray,
    random_at: np.ndarray,
    size: float,
    task_groups: List[Tuple[str, int, int]],
    pt_groups: List[Tuple[str, str, int, int]],
    task_strip: np.ndarray,
    strip_colors: List[str],
    prompt_to_task: Dict[str, str],
    prompt_colors: Dict[str, str],
    prompt_order: List[str],
    out_path: Path,
) -> None:
    cluster_ids = sorted(cluster_data.keys())
    k = len(cluster_ids)
    if dcd_at.shape[0] != k or random_at.shape[0] != k:
        raise ValueError(
            f"k mismatch: bars={k}, dcd={dcd_at.shape[0]}, random={random_at.shape[0]}"
        )

    strip_cmap = ListedColormap(strip_colors)
    cmap, vmin, vmax = "YlGnBu", 0.0, 1.0

    fig = plt.figure(figsize=(14.0, 5.6))

    # Geometry — manual placement so the strip sits exactly above the heatmaps
    # and the bar panel shares the same vertical extent (heat_bottom..heat_top).
    a_l, a_r = 0.050, 0.250        # bars (~20% width)
    b_l, b_r = 0.300, 0.585        # DCD heatmap
    c_l, c_r = 0.620, 0.905        # Random heatmap
    cb_l     = 0.920

    heat_bottom, heat_h = 0.180, 0.620
    strip_bottom, strip_h = heat_bottom + heat_h + 0.005, 0.024
    # Panel-title baseline (figure coords) — sits above the task-name labels.
    panel_title_y = 0.965

    ax_bar = fig.add_axes([a_l, heat_bottom, a_r - a_l, heat_h])
    ax_sB  = fig.add_axes([b_l, strip_bottom, b_r - b_l, strip_h])
    ax_sC  = fig.add_axes([c_l, strip_bottom, c_r - c_l, strip_h])
    ax_B   = fig.add_axes([b_l, heat_bottom, b_r - b_l, heat_h])
    ax_C   = fig.add_axes([c_l, heat_bottom, c_r - c_l, heat_h])
    ax_cb  = fig.add_axes([cb_l, heat_bottom, 0.012, heat_h])

    n_tasks = len(task_groups)
    _draw_bars(ax_bar, cluster_data, cluster_ids, prompt_order, prompt_colors, prompt_to_task)
    _draw_strip(ax_sB, task_strip, strip_cmap, n_tasks, pt_groups, task_groups)
    _draw_strip(ax_sC, task_strip, strip_cmap, n_tasks, pt_groups, task_groups)
    _draw_heatmap(ax_B, dcd_at, cmap, vmin, vmax, pt_groups, task_groups)
    im = _draw_heatmap(ax_C, random_at, cmap, vmin, vmax, pt_groups, task_groups)

    cb = fig.colorbar(im, cax=ax_cb, extend="both")
    cb.set_label(f"Per-example faithfulness (size={size:g})", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    panel_title_kwargs = dict(
        y=panel_title_y, ha="center", va="bottom",
        fontsize=13, fontweight="bold",
    )
    fig.text(x=(a_l + a_r) / 2, s="(a) Cluster composition", **panel_title_kwargs)
    fig.text(x=(b_l + b_r) / 2, s="(b) DCD circuits",        **panel_title_kwargs)
    fig.text(x=(c_l + c_r) / 2, s="(c) Random $k$-split",    **panel_title_kwargs)

    _draw_legend(fig, prompt_order, prompt_colors, prompt_to_task)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(_pdf_sibling(out_path))
    plt.close(fig)
    print(f"Saved {out_path} (+ .pdf)")


def plot_dcd_figure(
    cluster_data: Dict[int, pd.DataFrame],
    dcd_at: np.ndarray,
    size: float,
    task_groups: List[Tuple[str, int, int]],
    pt_groups: List[Tuple[str, str, int, int]],
    task_strip: np.ndarray,
    strip_colors: List[str],
    prompt_to_task: Dict[str, str],
    prompt_colors: Dict[str, str],
    prompt_order: List[str],
    out_path: Path,
) -> None:
    """Two-panel figure: cluster composition (a) + DCD per-example faithfulness (b).

    Panel B gets ~70% of figure width (vs ~28% in the 3-panel layout) so the
    prompt-type strip labels fit horizontally without overlap — needed when
    the test set spans many prompt types (e.g. all-tasks, llama).
    """
    cluster_ids = sorted(cluster_data.keys())
    if dcd_at.shape[0] != len(cluster_ids):
        raise ValueError(
            f"k mismatch: bars={len(cluster_ids)}, dcd={dcd_at.shape[0]}"
        )

    strip_cmap = ListedColormap(strip_colors)
    cmap, vmin, vmax = "YlGnBu", 0.0, 1.0

    fig = plt.figure(figsize=(14.0, 5.6))

    a_l, a_r = 0.050, 0.230
    b_l, b_r = 0.295, 0.910
    cb_l     = 0.925

    heat_bottom, heat_h = 0.180, 0.620
    strip_bottom, strip_h = heat_bottom + heat_h + 0.005, 0.024
    panel_title_y = 0.965

    ax_bar = fig.add_axes([a_l, heat_bottom, a_r - a_l, heat_h])
    ax_sB  = fig.add_axes([b_l, strip_bottom, b_r - b_l, strip_h])
    ax_B   = fig.add_axes([b_l, heat_bottom, b_r - b_l, heat_h])
    ax_cb  = fig.add_axes([cb_l, heat_bottom, 0.012, heat_h])

    n_tasks = len(task_groups)
    _draw_bars(ax_bar, cluster_data, cluster_ids, prompt_order, prompt_colors, prompt_to_task)
    _draw_strip(ax_sB, task_strip, strip_cmap, n_tasks, pt_groups, task_groups)
    im = _draw_heatmap(ax_B, dcd_at, cmap, vmin, vmax, pt_groups, task_groups)

    cb = fig.colorbar(im, cax=ax_cb, extend="both")
    cb.set_label(f"Per-example faithfulness (size={size:g})", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    panel_title_kwargs = dict(
        y=panel_title_y, ha="center", va="bottom",
        fontsize=13, fontweight="bold",
    )
    fig.text(x=(a_l + a_r) / 2, s="(a) Cluster composition", **panel_title_kwargs)
    fig.text(x=(b_l + b_r) / 2, s="(b) DCD circuits",        **panel_title_kwargs)

    _draw_legend(fig, prompt_order, prompt_colors, prompt_to_task)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(_pdf_sibling(out_path))
    plt.close(fig)
    print(f"Saved {out_path} (+ .pdf)")


def plot_random_figure(
    random_at: np.ndarray,
    size: float,
    task_groups: List[Tuple[str, int, int]],
    pt_groups: List[Tuple[str, str, int, int]],
    task_strip: np.ndarray,
    strip_colors: List[str],
    prompt_to_task: Dict[str, str],
    prompt_colors: Dict[str, str],
    prompt_order: List[str],
    out_path: Path,
) -> None:
    """Single-panel figure: random k-split per-example faithfulness baseline.

    Companion to plot_dcd_figure — same task-strip layout, colorbar range and
    legend so the two figures read as a paired comparison.
    """
    k = random_at.shape[0]

    strip_cmap = ListedColormap(strip_colors)
    cmap, vmin, vmax = "YlGnBu", 0.0, 1.0

    fig = plt.figure(figsize=(14.0, 4.6))

    c_l, c_r = 0.075, 0.910
    cb_l     = 0.925

    heat_bottom, heat_h = 0.220, 0.580
    strip_bottom, strip_h = heat_bottom + heat_h + 0.005, 0.030
    panel_title_y = 0.955

    ax_sC = fig.add_axes([c_l, strip_bottom, c_r - c_l, strip_h])
    ax_C  = fig.add_axes([c_l, heat_bottom, c_r - c_l, heat_h])
    ax_cb = fig.add_axes([cb_l, heat_bottom, 0.012, heat_h])

    n_tasks = len(task_groups)
    _draw_strip(ax_sC, task_strip, strip_cmap, n_tasks, pt_groups, task_groups)
    im = _draw_heatmap(ax_C, random_at, cmap, vmin, vmax, pt_groups, task_groups)

    # Add y-axis labels (no panel A in this figure to carry them)
    ax_C.set_yticks(range(k))
    ax_C.set_yticklabels([f"$r_{{{i + 1}}}$" for i in range(k)])
    ax_C.set_ylabel("Random partition")

    cb = fig.colorbar(im, cax=ax_cb, extend="both")
    cb.set_label(f"Per-example faithfulness (size={size:g})", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.text(
        x=(c_l + c_r) / 2, y=panel_title_y, s="Random $k$-split",
        ha="center", va="bottom", fontsize=13, fontweight="bold",
    )

    _draw_legend(fig, prompt_order, prompt_colors, prompt_to_task)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(_pdf_sibling(out_path))
    plt.close(fig)
    print(f"Saved {out_path} (+ .pdf)")


# ── Main ─────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", default="kmeans-pca")
    p.add_argument("--size", type=float, default=0.05,
                   help="Circuit size at which to slice per-example faithfulness.")
    p.add_argument("--eval-dirname", default="evaluation",
                   help="Subdirectory under analysis/ holding the per-example npz files.")
    p.add_argument("--output", default=None,
                   help="Optional output path stem (no extension). "
                        "Default: <results>/analysis/circuit_specialization_3panel_<size>.png")
    p.add_argument("--split", action="store_true",
                   help="Save two figures (DCD: a+b, Random: c) instead of the "
                        "combined 3-panel figure. Use when the test set spans "
                        "many prompt types and strip labels overlap.")
    args = p.parse_args()

    config = OmegaConf.load(args.config)
    results_root = _results_root(config)
    analysis_dir = results_root / "analysis"

    # Cluster CSVs (Panel A)
    cluster_dir = results_root / "circuits" / "created" / "dcd" / args.method
    cluster_data = _load_cluster_csvs(cluster_dir)

    # Per-example faithfulness npz (Panels B, C)
    eval_dir = analysis_dir / args.eval_dirname
    dcd_npz    = eval_dir / "dcd" / f"{args.method}.npz"
    random_npz = eval_dir / "random_k_split.npz"
    dcd_data, rand_data = np.load(dcd_npz), np.load(random_npz)

    sizes = dcd_data["sizes"]
    if not np.array_equal(sizes, rand_data["sizes"]):
        raise ValueError("DCD/Random size grids differ")
    matches = np.where(np.isclose(sizes, args.size))[0]
    if len(matches) == 0:
        raise ValueError(f"size={args.size} not in evaluation grid {list(sizes)}")
    size_idx = int(matches[0])

    dcd_faith    = dcd_data["faith"][:, size_idx, :]    # (k, n_test)
    random_faith = rand_data["faith"][:, size_idx, :]   # (k, n_test)

    # Reorder npz rows by integer id parsed from circuit_files. evaluate_dcd /
    # evaluate_random_k_split saved rows in lex order (cluster_1, cluster_10,
    # cluster_11, cluster_2, ...) which would misalign with panel (a)'s
    # numerically-sorted cluster_ids.
    def _row_order(circuit_files: np.ndarray) -> np.ndarray:
        ids = np.array([
            int(re.search(r"_(\d+)", str(f)).group(1)) for f in circuit_files
        ])
        return np.argsort(ids, kind="stable")

    dcd_faith    = dcd_faith[_row_order(dcd_data["circuit_files"])]
    random_faith = random_faith[_row_order(rand_data["circuit_files"])]

    # Sort test examples by (task, prompt_type)
    test_df = pd.read_csv(Path(config.paths.output_dir) / "test.csv")
    pt_to_task = _build_prompt_type_to_task(config)
    sort_idx, task_groups, test_sorted = _sort_test(test_df, pt_to_task)
    dcd_sorted    = dcd_faith[:, sort_idx]
    random_sorted = random_faith[:, sort_idx]

    pt_groups: List[Tuple[str, str, int, int]] = []
    for (task, pt), grp in test_sorted.groupby(["task_name", "prompt_type"], sort=False):
        pt_groups.append(
            (str(task), str(pt), int(grp.index.min()), int(grp.index.max()) + 1)
        )

    task_order = {t: i for i, (t, _, _) in enumerate(task_groups)}
    task_strip = np.array([task_order[t] for t in test_sorted["task_name"]])[np.newaxis, :]
    strip_colors = [_STRIP_PALETTE[i % len(_STRIP_PALETTE)] for i in range(len(task_groups))]

    # Bar chart colors / order — driven by the same config
    task_prompt_map = {t: list(cfg.prompt_types) for t, cfg in config.data.tasks.items()}
    prompt_to_task, prompt_colors, prompt_order = _generate_prompt_colors(task_prompt_map)

    if args.split:
        if args.output is None:
            dcd_out    = analysis_dir / f"circuit_specialization_dcd_{args.size:g}.png"
            random_out = analysis_dir / f"circuit_specialization_random_{args.size:g}.png"
        else:
            base = _strip_image_ext(args.output)
            dcd_out    = Path(f"{base}_dcd.png")
            random_out = Path(f"{base}_random.png")

        plot_dcd_figure(
            cluster_data=cluster_data,
            dcd_at=dcd_sorted,
            size=args.size,
            task_groups=task_groups,
            pt_groups=pt_groups,
            task_strip=task_strip,
            strip_colors=strip_colors,
            prompt_to_task=prompt_to_task,
            prompt_colors=prompt_colors,
            prompt_order=prompt_order,
            out_path=dcd_out,
        )
        plot_random_figure(
            random_at=random_sorted,
            size=args.size,
            task_groups=task_groups,
            pt_groups=pt_groups,
            task_strip=task_strip,
            strip_colors=strip_colors,
            prompt_to_task=prompt_to_task,
            prompt_colors=prompt_colors,
            prompt_order=prompt_order,
            out_path=random_out,
        )
    else:
        if args.output is None:
            out = analysis_dir / f"circuit_specialization_3panel_{args.size:g}.png"
        else:
            out = Path(_strip_image_ext(args.output) + ".png")

        plot_three_panel(
            cluster_data=cluster_data,
            dcd_at=dcd_sorted,
            random_at=random_sorted,
            size=args.size,
            task_groups=task_groups,
            pt_groups=pt_groups,
            task_strip=task_strip,
            strip_colors=strip_colors,
            prompt_to_task=prompt_to_task,
            prompt_colors=prompt_colors,
            prompt_order=prompt_order,
            out_path=out,
        )


if __name__ == "__main__":
    main()
