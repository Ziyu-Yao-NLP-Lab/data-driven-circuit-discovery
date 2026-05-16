import argparse
import gc
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

# Repo root so `src` and `eap` imports work when run as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.dcd.clustering import (
    NumpyJSONEncoder,
    binarize_matrix,
    compute_gap_statistic,
    find_elbow_k,
    get_hyperparameter_combos,
    run_clustering_method,
    save_cluster_examples,
    save_clustering_results,
    sparsify_matrix,
)
from sklearn.decomposition import PCA, TruncatedSVD
# find_elbow_k_hierarchical is not needed in select — elbow_k is pre-computed
# during the cluster stage and stored in results.json
from src.utils.graph_utils import build_edge_index, extract_edge_names, extract_edge_scores, load_scores_fast


def preprocess(config: OmegaConf, results_root: Path, force: bool) -> None:
    """Stack per-example edge scores into a matrix and save to disk.

    Args:
        config: Loaded experiment config.
        results_root: Root results directory for this task/model combination.
        force: Overwrite existing edge_scores.npy if True.
    """
    circuits_dir = results_root / "circuits"
    raw_dir = circuits_dir / "raw_edge_scores"
    scores_path = circuits_dir / "edge_scores.npy"
    names_path = circuits_dir / "edge_names.npy"

    method = config.attribution.method
    pattern = re.compile(rf"^train-(\d+)-{re.escape(method)}\.pt$")
    matched = []
    for p in raw_dir.glob("*.pt"):
        m = pattern.match(p.name)
        if m:
            matched.append((int(m.group(1)), p))
    matched.sort(key=lambda x: x[0])
    pt_files = [p for _, p in matched]

    if scores_path.exists() and not force:
        cached_rows = np.load(scores_path, mmap_mode="r").shape[0]
        if cached_rows == len(pt_files):
            print(f"edge_scores.npy already exists at {scores_path} "
                  f"({cached_rows} rows). Skipping. Use --force to regenerate.")
            return
        print(
            f"WARNING: edge_scores.npy has {cached_rows} rows but "
            f"{len(pt_files)} .pt files found. Cache is stale — regenerating."
        )

    if not pt_files:
        raise FileNotFoundError(
            f"No files matching 'train-{{i}}-{method}.pt' found in {raw_dir}"
        )

    # Validate against train.csv so any missing files (gaps or tail truncation
    # from an interrupted attribution run) are caught before building the matrix.
    results_train_csv = results_root / "data" / "train.csv"
    train_csv_path = (
        results_train_csv
        if results_train_csv.exists()
        else Path(config.paths.output_dir) / "train.csv"
    )
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"train.csv not found at {train_csv_path}. "
            "Run 01_create_data.py and 02_run_attribution.py before preprocessing."
        )
    n_train = len(pd.read_csv(train_csv_path))
    missing_from_csv = [
        i for i in range(n_train)
        if not (raw_dir / f"train-{i}-{method}.pt").exists()
    ]
    max_allowed_missing = max(1, int(n_train * 0.01))
    if len(missing_from_csv) > max_allowed_missing:
        raise ValueError(
            f"Expected {n_train} files per train.csv but "
            f"{len(missing_from_csv)} are missing in {raw_dir}.\n"
            f"First missing index: {missing_from_csv[0]} "
            f"(expected train-{missing_from_csv[0]}-{method}.pt).\n"
            "Run 02_run_attribution.py to completion before proceeding."
        )
    elif missing_from_csv:
        print(
            f"WARNING: {len(missing_from_csv)} of {n_train} files are missing "
            f"(likely NaN-skipped examples): {missing_from_csv}. "
            f"Proceeding with {n_train - len(missing_from_csv)} examples."
        )

    print(f"Found {len(pt_files)} files matching 'train-{{i}}-{method}.pt' in {raw_dir}")

    print("Building edge index from reference graph (one-time Graph.from_model call)...")
    edge_names, rows, cols = build_edge_index(str(pt_files[0]))
    print(f"Edge index built: {len(edge_names)} edges.")

    matrix = np.empty((len(pt_files), len(edge_names)), dtype=np.float32)
    for i, pt_path in enumerate(tqdm(pt_files, desc="Extracting edge scores")):
        matrix[i] = load_scores_fast(str(pt_path), rows, cols)
    print(f"Edge score matrix shape: {matrix.shape}")

    # Save the train.csv index each matrix row corresponds to, so downstream
    # code can correctly map rows back to train examples even when some
    # indices are missing (e.g., NaN-skipped during attribution).
    train_indices = np.array([idx for idx, _ in matched], dtype=np.int64)

    indices_path = circuits_dir / "train_indices.npy"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    np.save(scores_path, matrix)
    np.save(names_path, np.array(edge_names))
    np.save(indices_path, train_indices)
    print(f"Saved edge_scores.npy  → {scores_path}")
    print(f"Saved edge_names.npy   → {names_path}  ({len(edge_names)} edges)")
    print(f"Saved train_indices.npy → {indices_path}  ({len(train_indices)} indices)")


def cluster(
    config: OmegaConf,
    results_root: Path,
    model_config_path: str,
    clustering_config_path: str,
    force: bool = False,
    methods_filter: Optional[set] = None,
    cli_binarize: bool = False,
    cli_sparsify: bool = False,
) -> None:
    """Run the full hyperparameter grid of clustering methods.

    For each method in the clustering config and each combination of its
    list-valued hyperparameters, applies the requested preprocessing (raw,
    binary, or sparse), runs clustering, saves results and per-cluster
    example CSVs, and copies both configs into the combo directory.

    Preprocessing is resolved per method:
      - If --binarize is passed, every method uses "binary" (overrides yaml).
      - Else if --sparsify is passed, every method uses "sparse" (overrides yaml).
      - Otherwise, each method uses its yaml ``preprocessing`` field,
        defaulting to "raw".

    Args:
        config: Merged model + clustering config.
        results_root: Root results directory for this task/model combination.
        model_config_path: Path to the model-specific YAML (for copying).
        clustering_config_path: Path to clustering_grid.yaml (for copying).
        force: Re-run and overwrite existing combo results if True.
        methods_filter: Optional set of method names to run (others skipped).
        cli_binarize: Force preprocessing="binary" for all methods.
        cli_sparsify: Force preprocessing="sparse" for all methods.
    """
    if cli_binarize and cli_sparsify:
        raise ValueError("--binarize and --sparsify are mutually exclusive")
    cli_override: Optional[str] = None
    if cli_binarize:
        cli_override = "binary"
    elif cli_sparsify:
        cli_override = "sparse"
    scores_path = results_root / "circuits" / "edge_scores.npy"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"edge_scores.npy not found at {scores_path}. "
            "Run --stage preprocess first."
        )

    edge_scores = np.load(scores_path, mmap_mode='r')
    print(f"Loaded edge scores (mmap): {edge_scores.shape}")

    # Load the explicit train.csv → matrix-row mapping saved by preprocess so
    # we correctly skip NaN-dropped indices instead of naively truncating.
    indices_path = results_root / "circuits" / "train_indices.npy"
    if indices_path.exists():
        train_indices = np.load(indices_path)
        if len(train_indices) != len(edge_scores):
            raise ValueError(
                f"train_indices.npy has {len(train_indices)} entries but "
                f"edge_scores.npy has {len(edge_scores)} rows. Rerun "
                f"--stage preprocess --force to regenerate both."
            )
    else:
        # Back-compat: older preprocess runs didn't save indices. Assume
        # the matrix is aligned with the first N rows of train.csv.
        train_indices = np.arange(len(edge_scores), dtype=np.int64)
        print("WARNING: train_indices.npy not found — assuming contiguous 0..N-1 alignment. "
              "Rerun --stage preprocess --force if any attribution indices were skipped.")

    train_head_csv = results_root / "train_head.csv"
    full_train_csv = Path(config.paths.output_dir) / "train.csv"
    train_candidates = [train_head_csv, full_train_csv]
    train_data = None
    train_csv = None
    n_scores = len(edge_scores)
    required_max = int(train_indices.max()) + 1 if len(train_indices) else 0
    for candidate in train_candidates:
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate).reset_index(drop=True)
        if len(df) < required_max:
            continue
        # Select only the rows that actually have edge scores, preserving order.
        train_data = df.iloc[train_indices].reset_index(drop=True)
        train_csv = candidate
        break

    if train_data is None:
        raise FileNotFoundError(
            f"No training CSV with >= {required_max} rows found. Expected one of: "
            + ", ".join(str(p) for p in train_candidates)
        )

    print(f"Loaded train data: {len(train_data)} rows from {train_csv} "
          f"(selected via train_indices.npy, skipping {required_max - len(train_data)} missing)")

    clustering_root = results_root / "clustering"
    max_clusters: int = config.clustering.max_clusters

    # Pre-collect every (method_cfg, combo) pair grouped by (preprocessing, gamma)
    # so we preprocess the edge scores once per group and free the result before
    # the next one. For raw preprocessing, gamma is ignored — all raw runs share
    # a single group keyed on None.
    methods_list = list(config.clustering.methods)
    if methods_filter is not None:
        available = {OmegaConf.to_container(m, resolve=True)["name"] for m in methods_list}
        missing = methods_filter - available
        if missing:
            raise ValueError(
                f"--methods references unknown method(s): {sorted(missing)}. "
                f"Available in clustering_grid.yaml: {sorted(available)}"
            )
        methods_list = [
            m for m in methods_list
            if OmegaConf.to_container(m, resolve=True)["name"] in methods_filter
        ]
        print(f"Filtered to {len(methods_list)} method(s): {sorted(methods_filter)}")

    group_to_runs: Dict[Tuple[str, Optional[float]], list] = defaultdict(list)
    for method_cfg_raw in methods_list:
        method_cfg = OmegaConf.to_container(method_cfg_raw, resolve=True)
        method_name: str = method_cfg["name"]
        method_preprocessing: str = cli_override or method_cfg.get("preprocessing", "raw")
        if method_preprocessing not in ("raw", "binary", "sparse"):
            raise ValueError(
                f"Method {method_name!r} has invalid preprocessing "
                f"{method_preprocessing!r}; must be 'raw', 'binary', or 'sparse'."
            )
        for combo in get_hyperparameter_combos(method_cfg):
            gamma_key: Optional[float] = (
                combo.get("sparsity_gamma") if method_preprocessing != "raw" else None
            )
            group_to_runs[(method_preprocessing, gamma_key)].append(
                (method_cfg, combo, method_name, method_preprocessing)
            )

    total_combos = sum(len(v) for v in group_to_runs.values())
    print(
        f"\n{len(group_to_runs)} distinct (preprocessing, gamma) group(s), "
        f"{total_combos} combos total"
    )
    if cli_override is not None:
        print(f"CLI override: forcing preprocessing='{cli_override}' for every method")

    def _group_sort_key(k: Tuple[str, Optional[float]]) -> Tuple[str, float]:
        return (k[0], -1.0 if k[1] is None else float(k[1]))

    for group_key in sorted(group_to_runs.keys(), key=_group_sort_key):
        preprocessing, gamma_key = group_key
        runs = group_to_runs[group_key]
        label = (
            f"preprocessing={preprocessing}, gamma={gamma_key}"
            if gamma_key is not None
            else f"preprocessing={preprocessing}"
        )
        print(f"\n--- {label}  ({len(runs)} combos) ---")

        if preprocessing == "binary":
            clustering_input = binarize_matrix(edge_scores, gamma_key)
        elif preprocessing == "sparse":
            clustering_input = sparsify_matrix(edge_scores, gamma_key)
        else:
            # Raw: pass the edge scores through without any preprocessing.
            clustering_input = edge_scores

        for method_cfg, combo, method_name, method_preproc in tqdm(runs, desc=label):
            combo_name: str = combo["combo_name"]
            combo_dir = clustering_root / method_name / combo_name

            # Check for any existing results.json inside this combo dir
            existing = list(combo_dir.rglob("results.json")) if combo_dir.exists() else []
            if existing and not force:
                print(f"  combo: {combo_name}  [skipping — results.json exists, use --force to rerun]")
                continue

            combo_dir.mkdir(parents=True, exist_ok=True)
            print(f"  combo: {combo_name}")

            method_run_cfg = {
                "name": method_name,
                "reduction": combo.get("reduction"),
                "n_components": combo.get("n_components"),
                "linkage": combo.get("linkage", "average"),
                "random_state": combo.get("random_state", 42),
                "preprocessing": method_preproc,
            }
            global_cfg = {
                "max_clusters": max_clusters,
                "distance_metric": combo.get("distance_metric"),
                "sparsity_gamma": combo.get("sparsity_gamma"),
            }

            results = run_clustering_method(clustering_input, method_run_cfg, global_cfg)

            # Remap representative indices from matrix row index to the
            # original train.csv index, so downstream consumers (stage 04)
            # can look up raw_edge_scores/train-{idx}-*.pt directly.
            for result_key, k_results in results.items():
                for k, k_data in k_results.items():
                    if not isinstance(k_data, dict):
                        continue  # skip scalar entries like elbow_k
                    reps = k_data.get("representatives")
                    if reps:
                        k_data["representatives"] = {
                            int(cid): int(train_indices[row_idx])
                            for cid, row_idx in reps.items()
                        }

            # Save cluster example CSVs before save_clustering_results pops groups
            for result_key, k_results in results.items():
                result_save_dir = str(combo_dir / result_key)
                for k, k_data in k_results.items():
                    if not isinstance(k_data, dict):
                        continue  # skip scalar entries like elbow_k
                    save_cluster_examples(k_data["groups"], train_data, result_save_dir, k)

            save_clustering_results(results, str(combo_dir))

            shutil.copy(model_config_path, combo_dir)
            shutil.copy(clustering_config_path, combo_dir)

        # Free the preprocessed matrix before moving to the next group.
        # (For raw, clustering_input is an alias of edge_scores — dropping
        # the local reference is harmless.)
        del clustering_input
        gc.collect()

    print(f"\nClustering complete. Results under {clustering_root}/")


def _parse_result_key_reduction(result_key: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract reduction method and n_components from a result_key string.

    Result keys follow the pattern: {method}-{reduction}-{n_components}
    e.g. "kmeans-svd-truncated_svd-20", "kmeans-pca-pca-15",
         "hierarchical-agglomerative" (no reduction).

    Returns:
        (reduction_method, n_components) or (None, None) if no reduction.
    """
    import re as _re
    m = _re.search(r"(truncated_svd|pca)-(\d+)$", result_key)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def _parse_gamma_from_combo(combo_name: str) -> float:
    """Extract sparsity gamma from a combo name like 'gamma0.99_ncomp20'."""
    for part in combo_name.split("_"):
        if part.startswith("gamma"):
            try:
                return float(part[5:])
            except ValueError:
                pass
    return 0.99


def select(
    config: OmegaConf,
    results_root: Path,
    force: bool = False,
    n_refs: int = 20,
) -> None:
    """Select the best hyperparameter combination and k for each method.

    Reads all results.json files under the clustering directory. For each
    method, picks the combo and k with the highest silhouette score, via the
    elbow method on inertias (k-means/hierarchical), and via the gap statistic
    (methods with dimensionality reduction only).

    Saves best_configs.json to the clustering root directory.

    Args:
        config: Loaded experiment config (used only for results_root derivation).
        results_root: Root results directory for this task/model combination.
        force: Overwrite best_configs.json if it already exists.
        n_refs: Number of uniform reference datasets for the gap statistic.
    """
    clustering_root = results_root / "clustering"
    if not clustering_root.exists():
        raise FileNotFoundError(
            f"Clustering directory not found: {clustering_root}. "
            "Run --stage cluster first."
        )

    scores_path = results_root / "circuits" / "edge_scores.npy"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"edge_scores.npy not found at {scores_path}. "
            "Run --stage preprocess first."
        )
    edge_scores = np.load(scores_path)
    expected_rows = edge_scores.shape[0]
    print(f"Loaded edge_scores.npy: {edge_scores.shape}")

    # Build valid_combos from config if clustering.methods is present
    # (i.e. --clustering-config was provided for the select stage).
    # This prevents stale on-disk results from previous grid runs from
    # polluting any metric computation.
    valid_combos: Optional[set] = None
    if OmegaConf.select(config, "clustering.methods") is not None:
        valid_combos = set()
        for method_cfg_raw in config.clustering.methods:
            method_cfg = OmegaConf.to_container(method_cfg_raw, resolve=True)
            method_name_cfg: str = method_cfg["name"]
            for combo in get_hyperparameter_combos(method_cfg):
                valid_combos.add((method_name_cfg, combo["combo_name"]))
        print(
            f"Clustering config loaded: restricting to {len(valid_combos)} combos "
            f"across {len(config.clustering.methods)} method(s)."
        )
    else:
        print("No clustering config provided: considering all on-disk results.")

    # Collect all results grouped by method_name
    # Path structure: {method_name}/{combo_name}/{result_key}/results.json
    method_runs: dict = {}
    for results_json_path in sorted(clustering_root.rglob("results.json")):
        rel = results_json_path.relative_to(clustering_root)
        parts = rel.parts
        if len(parts) != 4:  # method / combo / result_key / results.json
            continue
        method_name, combo_name, result_key = parts[0], parts[1], parts[2]

        # Skip combos not in the current clustering grid (if config provided)
        if valid_combos is not None and (method_name, combo_name) not in valid_combos:
            continue

        result_dir_path = results_json_path.parent
        bad_npy = []
        for npy_path in sorted(result_dir_path.glob("groups_k*.npy")):
            nrows = np.load(npy_path, mmap_mode='r').shape[0]
            if nrows != expected_rows:
                bad_npy.append(f"{npy_path.name} ({nrows} rows)")
        if bad_npy:
            raise RuntimeError(
                f"Data integrity error in "
                f"{result_dir_path.relative_to(clustering_root)}: "
                f"expected {expected_rows} rows but found mismatches: "
                + ", ".join(bad_npy)
                + ". Delete the stale directory and rerun --stage cluster."
            )

        with open(results_json_path) as f:
            results_data = json.load(f)

        method_runs.setdefault(method_name, []).append({
            "combo_name": combo_name,
            "result_key": result_key,
            "result_dir": str(results_json_path.parent),
            "results": results_data,
        })

    if not method_runs:
        raise RuntimeError(
            f"No results.json files found under {clustering_root}. "
            "Run --stage cluster first."
        )

    best_configs: dict = {}
    for method_name, runs in sorted(method_runs.items()):
        is_kmeans = "kmeans" in method_name
        is_hierarchical = "hierarchical" in method_name

        best_sil = float("-inf")
        best_run = None
        best_k: int = 2

        for run in runs:
            for k_str, k_data in run["results"].items():
                if not k_str[0].isdigit():
                    continue  # skip non-k entries like "elbow_k"
                k = int(k_str.split("-")[0])
                sil = k_data.get("silhouette_score", float("-inf"))
                if sil > best_sil:
                    best_sil = sil
                    best_run = run
                    best_k = k

        if best_run is None:
            continue

        entry: dict = {
            "best_combo": best_run["combo_name"],
            "best_result_key": best_run["result_key"],
            "best_k": best_k,
            "best_silhouette": best_sil,
            "result_dir": best_run["result_dir"],
        }

        if is_kmeans or is_hierarchical:
            # Find the best combo/k using the elbow criterion.
            # k-means: elbow from inertia curve; hierarchical: elbow_k pre-computed
            # from dendrogram merge heights (agglomerative) or pseudo-inertia (divisive).
            # Among all combos, pick the one with the highest silhouette at its elbow k.
            best_elbow_sil = float("-inf")
            best_elbow_run = None
            best_elbow_k: int = 2

            for run in runs:
                if is_kmeans:
                    inertias: Dict[int, float] = {}
                    for k_str, k_data in run["results"].items():
                        if not k_str[0].isdigit():
                            continue
                        k = int(k_str.split("-")[0])
                        if "inertia" in k_data:
                            inertias[k] = k_data["inertia"]
                    if not inertias:
                        continue
                    elbow_k = find_elbow_k(inertias)
                else:
                    # elbow_k stored during cluster stage
                    elbow_k = run["results"].get("elbow_k")
                    if elbow_k is None:
                        continue

                elbow_sil = run["results"].get(
                    f"{elbow_k}-clusters", {}
                ).get("silhouette_score", float("-inf"))
                if elbow_sil > best_elbow_sil:
                    best_elbow_sil = elbow_sil
                    best_elbow_run = run
                    best_elbow_k = elbow_k

            if best_elbow_run is not None:
                entry["best_combo_elbow"] = best_elbow_run["combo_name"]
                entry["best_result_key_elbow"] = best_elbow_run["result_key"]
                entry["best_k_elbow"] = best_elbow_k
                entry["best_silhouette_elbow"] = best_elbow_sil
                entry["result_dir_elbow"] = best_elbow_run["result_dir"]

        # Gap statistic: only for runs that used dimensionality reduction
        # (gap statistic is intractable in the original ~1.5M-dim binary space).
        # Caches binarized and reduced data across combos to avoid redundant work.
        binary_cache: Dict[float, np.ndarray] = {}
        data_cache: Dict[tuple, np.ndarray] = {}

        best_gap_sil = float("-inf")
        best_gap_run = None
        best_gap_k: int = 2

        for run in runs:
            reduction, n_components = _parse_result_key_reduction(run["result_key"])
            if reduction is None:
                continue  # skip binary-space methods

            gamma = _parse_gamma_from_combo(run["combo_name"])
            cache_key = (gamma, reduction, n_components)

            if cache_key not in data_cache:
                if gamma not in binary_cache:
                    print(f"  Binarizing edge scores (gamma={gamma})...")
                    binary_cache[gamma] = binarize_matrix(edge_scores, gamma)
                binary = binary_cache[gamma]
                print(f"  Applying {reduction} (n_components={n_components}) "
                      f"for gap statistic...")
                if reduction == "truncated_svd":
                    reducer = TruncatedSVD(n_components=n_components, random_state=42)
                else:
                    reducer = PCA(n_components=n_components, random_state=42)
                data_cache[cache_key] = reducer.fit_transform(binary.astype(np.float32))

            clust_data = data_cache[cache_key]

            # Load all saved group assignments for this combo
            result_dir_path = Path(run["result_dir"])
            groups_by_k: Dict[int, np.ndarray] = {}
            for k_str in run["results"]:
                if not k_str[0].isdigit():
                    continue
                k = int(k_str.split("-")[0])
                npy_path = result_dir_path / f"groups_k{k}.npy"
                if npy_path.exists():
                    groups_by_k[k] = np.load(npy_path)

            if not groups_by_k:
                continue

            print(f"  Gap statistic: {method_name}/{run['combo_name']} "
                  f"(n_refs={n_refs})...")
            _, gap_k = compute_gap_statistic(
                clust_data, groups_by_k, n_refs=n_refs
            )

            gap_sil = run["results"].get(
                f"{gap_k}-clusters", {}
            ).get("silhouette_score", float("-inf"))

            if gap_sil > best_gap_sil:
                best_gap_sil = gap_sil
                best_gap_run = run
                best_gap_k = gap_k

        if best_gap_run is not None:
            entry["best_combo_gap"] = best_gap_run["combo_name"]
            entry["best_result_key_gap"] = best_gap_run["result_key"]
            entry["best_k_gap"] = best_gap_k
            entry["best_silhouette_gap"] = best_gap_sil
            entry["result_dir_gap"] = best_gap_run["result_dir"]

        best_configs[method_name] = entry
        elbow_suffix = (
            f", elbow_combo={entry.get('best_combo_elbow')}, "
            f"elbow_k={entry.get('best_k_elbow')}, "
            f"elbow_silhouette={entry.get('best_silhouette_elbow', float('-inf')):.4f}"
            if (is_kmeans or is_hierarchical) and "best_k_elbow" in entry
            else ""
        )
        print(
            f"  {method_name}: best_combo={entry['best_combo']}, "
            f"best_k={best_k}, silhouette={best_sil:.4f}"
            + elbow_suffix
        )

    # Build all_configs: one entry per method, all combos and k values with scores.
    all_configs: dict = {}
    for method_name, runs in sorted(method_runs.items()):
        combos: dict = {}
        for run in runs:
            combo_name = run["combo_name"]
            results_data = run["results"]

            k_results: dict = {}
            for k_str, k_data in results_data.items():
                if not k_str[0].isdigit():
                    continue  # skip scalar entries like "elbow_k"
                k = int(k_str.split("-")[0])
                k_results[str(k)] = dict(k_data)

            combo_entry: dict = {
                "result_dir": run["result_dir"],
                "k_results": k_results,
            }
            elbow_k = results_data.get("elbow_k")
            if elbow_k is not None:
                combo_entry["elbow_k"] = int(elbow_k)

            combos[combo_name] = combo_entry

        all_configs[method_name] = {"combos": combos}

    all_out_path = clustering_root / "all_configs.json"
    if all_out_path.exists() and not force:
        print(f"all_configs.json already exists at {all_out_path}. Skipping. "
              "Use --force to regenerate.")
    else:
        with open(all_out_path, "w") as f:
            json.dump(all_configs, f, indent=2, cls=NumpyJSONEncoder)
        print(f"Saved all_configs.json  → {all_out_path}")

    out_path = clustering_root / "best_configs.json"
    if out_path.exists() and not force:
        print(f"best_configs.json already exists at {out_path}. Skipping. "
              "Use --force to regenerate.")
        return
    with open(out_path, "w") as f:
        json.dump(best_configs, f, indent=2, cls=NumpyJSONEncoder)
    print(f"Saved best_configs.json → {out_path}")


def main() -> None:
    """Entry point for the DCD clustering pipeline."""
    parser = argparse.ArgumentParser(
        description="DCD stage 3: preprocess edge scores and cluster examples.",
    )
    parser.add_argument("--config", required=True, help="Path to model-specific YAML config")
    parser.add_argument(
        "--clustering-config",
        default="configs/dcd/clustering_grid.yaml",
        help="Path to shared clustering_grid.yaml (required for --stage cluster)",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preprocess", "cluster", "select"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing outputs for the selected stage "
             "(preprocess: edge_scores.npy; cluster: per-combo results; "
             "select: best_configs.json and all_configs.json)",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated list of method names to run (--stage cluster only). "
             "Default 'all' runs every method in clustering_grid.yaml.",
    )
    parser.add_argument(
        "--n-refs",
        type=int,
        default=20,
        help="Number of uniform reference datasets for the gap statistic "
             "(--stage select only, default: 20).",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        default=False,
        help="Binarize edge scores before clustering (cluster stage). "
             "Overrides per-method preprocessing in clustering_grid.yaml. "
             "Mutually exclusive with --sparsify.",
    )
    parser.add_argument(
        "--sparsify",
        action="store_true",
        default=False,
        help="Apply sparsity-gamma thresholding (keep raw magnitudes above "
             "threshold, zero the rest) before clustering. Overrides "
             "per-method preprocessing. Mutually exclusive with --binarize.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. clustering.max_clusters=20",
    )
    args = parser.parse_args()

    model_config = OmegaConf.load(args.config)

    if args.stage == "cluster":
        if args.clustering_config is None:
            parser.error("--clustering-config is required for --stage cluster")
        clustering_config = OmegaConf.load(args.clustering_config)
        config = OmegaConf.merge(model_config, clustering_config)
    elif args.stage == "select" and args.clustering_config is not None:
        clustering_config = OmegaConf.load(args.clustering_config)
        config = OmegaConf.merge(model_config, clustering_config)
    else:
        config = model_config

    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    task_name: str = config.data.task_name
    model_dir: str = Path(config.paths.output_dir).name
    results_root = Path(f"results/dcd/{task_name}/{model_dir}")

    if args.stage == "preprocess":
        preprocess(config, results_root, force=args.force)
    elif args.stage == "cluster":
        methods_filter: Optional[set] = None
        if args.methods and args.methods != "all":
            methods_filter = {m.strip() for m in args.methods.split(",") if m.strip()}
        cluster(
            config,
            results_root,
            args.config,
            args.clustering_config,
            force=args.force,
            methods_filter=methods_filter,
            cli_binarize=args.binarize,
            cli_sparsify=args.sparsify,
        )
    elif args.stage == "select":
        select(config, results_root, force=args.force, n_refs=args.n_refs)


if __name__ == "__main__":
    main()
