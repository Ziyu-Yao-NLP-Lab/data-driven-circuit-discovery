"""
Plot cluster composition by task and prompt type for the DCD paper.

Usage:
    python plot_cluster_composition.py \
        --cluster-dir path/to/kmeans-pca/ \
        --config path/to/gpt2.yaml \
        --output cluster_composition \
        --title "Cluster composition (k-means PCA, GPT-2, all-tasks)"

The cluster directory should contain CSV files named cluster_1.csv, cluster_2.csv, etc.
Each CSV must have a 'prompt_type' column.

The config YAML should have a data.tasks section mapping task names to prompt_types:
    data:
      tasks:
        ioi:
          prompt_types: ["mixed", "letter", "3-person"]
        sequence-completion:
          prompt_types: ["2-gram-symbolic", "3-gram-symbolic", "4-gram-symbolic"]
"""

import argparse
import glob
import re
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


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
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ── Color generation ─────────────────────────────────────────
def generate_task_colors(task_prompt_map):
    """
    Generate warm hues for the first task, cool hues for the second, etc.
    Returns {prompt_type: color} and {prompt_type: task_name}.
    """
    # Base palettes per task (up to 6 tasks supported)
    task_palettes = [
        # Warm reds (task 0)
        ["#8b2323", "#c44e52", "#e8927c", "#f0c0a0", "#f5dcc8", "#fae8dc"],
        # Cool blues (task 1)
        ["#1a5276", "#2980b9", "#4c9bd6", "#7ec4e8", "#b0ddf0", "#d4eef8"],
        # Greens (task 2)
        ["#1e6e3e", "#27ae60", "#52c77e", "#82d9a0", "#b2e8c4", "#d8f4e0"],
        # Purples (task 3)
        ["#4a235a", "#7d3c98", "#a569bd", "#c39bd3", "#d7bde2", "#ebdef0"],
        # Oranges (task 4)
        ["#7e3f00", "#d35400", "#e67e22", "#f0a04b", "#f5c27a", "#fae0b0"],
        # Teals (task 5)
        ["#0e4d4d", "#138d8d", "#1abc9c", "#48d1b5", "#82e0cd", "#bef0e5"],
    ]

    prompt_to_task = {}
    prompt_colors = {}
    prompt_order = []

    for task_idx, (task_name, prompt_types) in enumerate(task_prompt_map.items()):
        palette = task_palettes[task_idx % len(task_palettes)]
        for pt_idx, pt in enumerate(prompt_types):
            prompt_to_task[pt] = task_name
            prompt_colors[pt] = palette[min(pt_idx + 1, len(palette) - 1)]
            prompt_order.append(pt)

    return prompt_to_task, prompt_colors, prompt_order


def shorten_prompt_name(name):
    """Remove common suffixes for display."""
    aliases = {
        "comma-0-position": "8-comma",
        "2-color-box-comma": "2-color-box",
    }
    if name in aliases:
        return aliases[name]
    name = name.replace("-symbolic", "")
    return name


def shorten_task_name(name, max_len=15):
    """Shorten task name for bar annotation."""
    abbrevs = {
        "sequence-completion": "SC",
        "seq-completion": "SC",
        "ioi": "IOI",
        "arithmetic": "Arith",
        "entity-binding": "EB",
    }
    return abbrevs.get(name, name[:max_len].upper())


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot cluster composition by task and prompt type."
    )
    parser.add_argument(
        "--cluster-dir", required=True,
        help="Directory containing cluster_1.csv, cluster_2.csv, etc.",
    )
    parser.add_argument(
        "--config", required=True,
        help="YAML config file with data.tasks mapping task -> prompt_types.",
    )
    parser.add_argument(
        "--output", default="cluster_composition",
        help="Output filename stem (saves .pdf and .png).",
    )
    parser.add_argument(
        "--title", default=None,
        help="Figure title. If not set, auto-generated from config.",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    task_prompt_map = {}
    for task_name, task_cfg in config["data"]["tasks"].items():
        task_prompt_map[task_name] = task_cfg["prompt_types"]

    prompt_to_task, prompt_colors, prompt_order = generate_task_colors(task_prompt_map)

    # ── Load cluster CSVs ────────────────────────────────────
    cluster_dir = Path(args.cluster_dir)
    csv_files = sorted(glob.glob(str(cluster_dir / "cluster_*.csv")))

    # Extract cluster numbers and sort
    cluster_data = {}
    for f in csv_files:
        match = re.search(r"cluster_(\d+)\.csv", f)
        if match:
            idx = int(match.group(1))
            cluster_data[idx] = pd.read_csv(f)

    if not cluster_data:
        raise FileNotFoundError(f"No cluster_*.csv files found in {cluster_dir}")

    n_clusters = len(cluster_data)
    cluster_ids = sorted(cluster_data.keys())
    print(f"Found {n_clusters} clusters: {cluster_ids}")

    # Print summary
    for idx in cluster_ids:
        df = cluster_data[idx]
        counts = df["prompt_type"].value_counts()
        print(f"  C{idx} (n={len(df)}): {dict(counts)}")

    # ── Plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(3.5, 0.7 * n_clusters + 1.5), 2.8))

    x_pos = np.arange(n_clusters)
    bottoms = np.zeros(n_clusters)

    for pt in prompt_order:
        task = prompt_to_task[pt]
        display = f"{task} / {shorten_prompt_name(pt)}"
        vals = np.array([
            cluster_data[idx]["prompt_type"].value_counts().get(pt, 0)
            for idx in cluster_ids
        ])
        ax.bar(
            x_pos, vals, bottom=bottoms,
            label=display, color=prompt_colors[pt],
            edgecolor="white", linewidth=0.5,
        )
        bottoms += vals

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of examples")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$C_{{{i}}}$" for i in cluster_ids])

    # Title (pass --title "" to suppress)
    if args.title is not None:
        ax.set_title(args.title)
    else:
        model = config.get("model", {}).get("name", "")
        task_name = config.get("data", {}).get("task_name", "")
        ax.set_title(f"Cluster composition ({model}, {task_name})")

    # Legend
    ax.legend(
        fontsize=5.5, loc="upper right", ncol=2,
        framealpha=0.9, columnspacing=1.0,
        title="task / prompt type", title_fontsize=5.5,
    )

    # Annotate dominant task above each bar (if ≥95% pure)
    for j, idx in enumerate(cluster_ids):
        counts = cluster_data[idx]["prompt_type"].value_counts()
        task_counts = {}
        for pt, n in counts.items():
            t = prompt_to_task.get(pt, "unknown")
            task_counts[t] = task_counts.get(t, 0) + n
        dominant = max(task_counts, key=task_counts.get)
        purity = task_counts[dominant] / sum(task_counts.values())
        if purity >= 0.95:
            label = shorten_task_name(dominant)
            # Color matches the task palette
            task_idx = list(task_prompt_map.keys()).index(dominant)
            color = WONG["vermillion"] if task_idx == 0 else WONG["blue"]
            ax.text(
                j, bottoms[j] + 2, label,
                ha="center", va="bottom", fontsize=5.5,
                fontweight="bold", color=color,
            )

    plt.tight_layout()
    plt.savefig(f"{args.output}.pdf")
    plt.savefig(f"{args.output}.png", dpi=300)

    figures_dir = Path("results/dcd/figures/analysis")
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.output).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mirror_pdf = figures_dir / f"{stem}_{timestamp}.pdf"
    mirror_png = figures_dir / f"{stem}_{timestamp}.png"
    plt.savefig(mirror_pdf)
    plt.savefig(mirror_png, dpi=300)
    plt.close()
    print(f"\nSaved: {args.output}.pdf and {args.output}.png")
    print(f"Also saved to: {mirror_pdf} and {mirror_png}")


if __name__ == "__main__":
    main()