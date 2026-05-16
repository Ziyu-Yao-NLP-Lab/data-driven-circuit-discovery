"""
Plot N-panel faithfulness curves for the DCD paper.

Usage:
    python plot_three_faith_curve.py \
        --files path/to/panel_a.json path/to/panel_b.json path/to/panel_c.json \
        --titles "(a) GPT-2 — All Tasks" "(b) Qwen-2.5 — All Tasks" "(c) Qwen-2.5 — Entity Binding" \
        --ylim=-0.15,1.15 --ylim=-0.15,1.15 --ylim=-0.15,1.4 \
        --eact 1 0 0  \
        --output faithfulness_3panel

Any number of panels >= 1 is accepted. If --titles/--ylim/--eact are omitted,
sensible defaults matching len(--files) are used. Use the --ylim=lo,hi (with =)
form because values starting with "-" are otherwise mistaken for flags.

Each JSON file should have the structure produced by evaluate_faithfulness.py, i.e.:
{
    "dcd": {
        "kmeans-pca": { "faithfulness": {"0.001": ..., ...} },
        "hierarchical-divisive-svd": { ... },
        ...
    },
    "single": { "faithfulness": {...} },
    "random": { "faithfulness": {...} },
    "random_k_split": { "faithfulness": {...} },
    "representative": { "faithfulness": {...} },
    "baseline": {
        "eap": { "faithfulness": {...} },
        "eact": { "faithfulness": {...} }   # optional
    }
}
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ── Wong (2011) colorblind-safe palette ──────────────────────
WONG = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "pink":       "#CC79A7",
    "sky":        "#56B4E9",
    "vermillion": "#D55E00",
    "yellow":     "#F0E442",
    "black":      "#000000",
}

# ── rcParams (NeurIPS style) ─────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 7,
    "legend.fontsize": 5.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ── Helpers ──────────────────────────────────────────────────
def get_faith(fdict, sizes):
    """Return faithfulness values; None for missing sizes."""
    return [fdict.get(s, None) for s in sizes]


def plot_line(ax, vals, **kwargs):
    """Plot skipping None entries (methods with fewer size points)."""
    xs, ys = [], []
    for i, v in enumerate(vals):
        if v is not None:
            xs.append(i)
            ys.append(v)
    ax.plot(xs, ys, **kwargs)


# ── Canonical name mapping ───────────────────────────────────
DCD_KEY_ALIASES = {
    "kmeans-pca":                      "kmeans-pca",
    "kmeans-svd":                      "kmeans-svd",
    "kmeans-pca-raw":                  "kmeans-pca-raw",
    "kmeans-svd-raw":                  "kmeans-svd-raw",
    "hierarchical-agglomerative":      "hier-agg",
    "hierarchical-agglomerative-svd":  "hier-agg-svd",
    "hierarchical-divisive":           "hier-div",
    "hierarchical-divisive-svd":       "hier-div-svd",
}


def load_panel_data(path):
    """Load a JSON results file and return a normalised dict of faithfulness curves."""
    with open(path) as f:
        raw = json.load(f)

    out = {}

    # DCD variants
    for key, fdata in raw.get("dcd", {}).items():
        canonical = DCD_KEY_ALIASES.get(key, key)
        out[canonical] = fdata["faithfulness"]

    # Top-level baselines
    for key in ("single", "random", "random_k_split", "representative"):
        if key in raw:
            out[key] = raw[key]["faithfulness"]

    # Nested baselines (eap, eact)
    for key, fdata in raw.get("baseline", {}).items():
        out[key] = fdata["faithfulness"]

    # Determine the size list from the first available curve
    first_curve = next(iter(out.values()))
    sizes = sorted(first_curve.keys(), key=float)
    out["_sizes"] = sizes

    return out


DCD_METHOD_LABELS = {
    "kmeans-pca":     "DCD (k-means PCA)",
    "kmeans-pca-raw": "DCD (k-means PCA, raw)",
    "kmeans-svd":     "DCD (k-means SVD)",
    "kmeans-svd-raw": "DCD (k-means SVD, raw)",
    "hier-agg":       "DCD (hier-agg)",
    "hier-agg-svd":   "DCD (hier-agg SVD)",
    "hier-div":       "DCD (hier-div)",
    "hier-div-svd":   "DCD (hier-div SVD)",
}


def make_panel(ax, data, title, show_ylabel=True, show_eact=False, dcd_method="kmeans-pca"):
    """Draw one panel of the faithfulness figure."""
    sizes = data["_sizes"]
    x_pos = np.arange(len(sizes))

    # ── DCD envelope ─────────────────────────────────────────
    dcd_keys = [
        k for k in [
            "kmeans-pca", "kmeans-svd", "kmeans-pca-raw", "kmeans-svd-raw",
            "hier-agg", "hier-agg-svd", "hier-div", "hier-div-svd",
        ]
        if k in data
    ]
    dcd_x, dcd_lo, dcd_hi = [], [], []
    for si, s in enumerate(sizes):
        vals = [data[k][s] for k in dcd_keys if s in data[k]]
        if vals:
            dcd_x.append(si)
            dcd_lo.append(min(vals))
            dcd_hi.append(max(vals))

    ax.fill_between(dcd_x, dcd_lo, dcd_hi, alpha=0.12, color=WONG["blue"])

    # ── DCD best ─────────────────────────────────────────────
    if dcd_method in data:
        dcd_show = dcd_method
    elif "kmeans-pca" in data:
        dcd_show = "kmeans-pca"
    else:
        dcd_show = dcd_keys[0] if dcd_keys else None
    if dcd_show:
        label = DCD_METHOD_LABELS.get(dcd_show, f"DCD ({dcd_show})")
        plot_line(
            ax, get_faith(data[dcd_show], sizes),
            color=WONG["blue"], marker="o", label=label, zorder=5,
        )

    # ── Splitting baseline ───────────────────────────────────
    if "random_k_split" in data:
        plot_line(
            ax, get_faith(data["random_k_split"], sizes),
            color=WONG["orange"], marker="s", linestyle="--", label="K-RANDOM",
        )

    # ── Representative baseline ──────────────────────────────
    if "representative" in data:
        plot_line(
            ax, get_faith(data["representative"], sizes),
            color=WONG["pink"], marker="D", linestyle="--", label="K-REPRESENTATIVE",
        )

    # ── Attribution baselines ────────────────────────────────
    if "single" in data:
        plot_line(
            ax, get_faith(data["single"], sizes),
            color=WONG["vermillion"], marker="^", linestyle="-.", label="EAP-IG (global)",
        )
    if "eap" in data:
        plot_line(
            ax, get_faith(data["eap"], sizes),
            color=WONG["green"], marker="v", linestyle="--", label="EAP (global)",
        )
    if show_eact and "eact" in data:
        plot_line(
            ax, get_faith(data["eact"], sizes),
            color=WONG["sky"], marker="<", linestyle="-.", label="E-Act (global)",
        )

    # ── Random edges ─────────────────────────────────────────
    if "random" in data:
        plot_line(
            ax, get_faith(data["random"], sizes),
            color=WONG["black"], marker="x", linestyle=":", alpha=0.5, label="Random edges",
        )

    # ── Formatting ───────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes, rotation=45, ha="right")
    ax.set_xlabel("Circuit size (fraction of edges)")
    if show_ylabel:
        ax.set_ylabel("Faithfulness")
    ax.set_title(title)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.3, lw=0.8)


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot N-panel faithfulness curves for the DCD paper."
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="Paths to one or more JSON faithfulness files (one per panel).",
    )
    parser.add_argument(
        "--titles", nargs="+", default=None,
        help="Panel titles (one per file). If omitted, panels are titled (a), (b), ...",
    )
    parser.add_argument(
        "--ylim", action="append", default=None, dest="ylim",
        help="Y-axis limits per panel as 'lo,hi'. Repeat once per panel "
             "(use '--ylim=lo,hi' so leading '-' isn't parsed as a flag). "
             "Pass once to apply to all panels. Default: -0.15,1.15 for all.",
    )
    parser.add_argument(
        "--eact", nargs="+", type=int, default=None,
        help="Whether to show EACT per panel (0 or 1). Default: 0 for all.",
    )
    parser.add_argument(
        "--legend", type=int, default=0,
        help="Which panel (0-indexed) gets the legend. Default: 0.",
    )
    parser.add_argument(
        "--output", default="results/dcd/figures/faithfulness_3panel.pdf",
        help="Output filename stem (saves .pdf and .png).",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=None,
        help="Optional list of circuit sizes to display (overrides per-panel sizes for consistency).",
    )
    parser.add_argument(
        "--panel-width", type=float, default=2.33,
        help="Width (inches) per panel. Default: 2.33 (matches old 7.0/3 figsize).",
    )
    parser.add_argument(
        "--panel-height", type=float, default=2.6,
        help="Height (inches) of the figure. Default: 2.6.",
    )
    parser.add_argument(
        "--dcd-method", default="kmeans-pca",
        choices=list(DCD_METHOD_LABELS.keys()),
        help="Which DCD variant to highlight as the main blue line. "
             "Falls back to kmeans-pca, then any available DCD key, if missing.",
    )
    args = parser.parse_args()

    n = len(args.files)

    # Resolve per-panel argument lists with defaults
    if args.titles is None:
        args.titles = [f"({chr(ord('a') + i)})" for i in range(n)]
    if len(args.titles) != n:
        parser.error(f"--titles has {len(args.titles)} entries, expected {n}")

    if args.ylim is None:
        args.ylim = ["-0.15,1.15"] * n
    elif len(args.ylim) == 1:
        args.ylim = args.ylim * n
    if len(args.ylim) != n:
        parser.error(f"--ylim repeated {len(args.ylim)} times, expected {n} or 1")

    if args.eact is None:
        args.eact = [0] * n
    elif len(args.eact) == 1:
        args.eact = args.eact * n
    if len(args.eact) != n:
        parser.error(f"--eact has {len(args.eact)} entries, expected {n} or 1")

    # Load data
    panels = [load_panel_data(p) for p in args.files]
    if args.sizes is not None:
        for panel in panels:
            panel["_sizes"] = list(args.sizes)
    ylims = [tuple(float(x) for x in yl.split(",")) for yl in args.ylim]

    # Build figure
    fig, axes = plt.subplots(
        1, n, figsize=(args.panel_width * n, args.panel_height), sharey=False,
    )
    if n == 1:
        axes = [axes]

    for i, (ax, data, title) in enumerate(zip(axes, panels, args.titles)):
        make_panel(
            ax, data, title,
            show_ylabel=(i == 0),
            show_eact=bool(args.eact[i]),
            dcd_method=args.dcd_method,
        )
        ax.set_ylim(*ylims[i])

    # Build a global legend (union of labels across all panels) and attach
    # it to the specified panel so EACT etc. show even if only one panel has it.
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h
    legend_idx = max(0, min(args.legend, n - 1))
    axes[legend_idx].legend(
        seen.values(), seen.keys(), loc="lower right", framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(f"{args.output}.pdf")
    plt.savefig(f"{args.output}.png", dpi=300)

    figures_dir = Path("results/dcd/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.output).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mirror_pdf = figures_dir / f"{stem}_{timestamp}.pdf"
    mirror_png = figures_dir / f"{stem}_{timestamp}.png"
    plt.savefig(mirror_pdf)
    plt.savefig(mirror_png, dpi=300)
    plt.close()
    print(f"Saved: {args.output}.pdf and {args.output}.png")
    print(f"Also saved to: {mirror_pdf} and {mirror_png}")


if __name__ == "__main__":
    main()