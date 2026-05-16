import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Repo root so `src` and `experiments` imports work when run as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from experiments.dcd.my_attribution import run_localization_dataset
from src.utils.general_utils import load_model
from src.utils.graph_utils import extract_edge_scores


# ---------------------- Helpers ----------------------

def _results_root(config: OmegaConf) -> Path:
    """Derive the results root directory from the config."""
    task_name: str = config.data.task_name
    model_dir: str = Path(config.paths.output_dir).name
    return Path(f"results/dcd/{task_name}/{model_dir}")


_BEST_K_KEYS = ["best_k_gap", "best_k_elbow", "best_k"]


def _resolve_best_k(entry: dict) -> int:
    """Return the maximum k across all available best-k keys in a best_configs entry.

    Considers best_k_gap, best_k_elbow, and best_k (whichever are present) and
    returns the largest value among them, so that the created circuits cover the
    broadest cluster structure suggested by any selection criterion.

    Args:
        entry: Single method entry from best_configs.json.

    Returns:
        Integer k value (maximum across all present best-k keys).

    Raises:
        KeyError: If none of the expected keys are present.
    """
    available = {k: int(entry[k]) for k in _BEST_K_KEYS if k in entry}
    if not available:
        raise KeyError(
            f"None of {_BEST_K_KEYS} found in best_configs entry. "
            f"Available keys: {list(entry.keys())}"
        )
    best_k = max(available.values())
    print(f"  Resolved best_k={best_k} from {available}")
    return best_k


def _load_best_configs(results_root: Path) -> dict:
    """Load best_configs.json written by 03_run_clustering.py --stage select.

    Args:
        results_root: Root results directory for this task/model combination.

    Returns:
        Parsed best_configs dict keyed by method name.
    """
    path = results_root / "clustering" / "best_configs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"best_configs.json not found at {path}. "
            "Run 03_run_clustering.py --stage select first."
        )
    with open(path) as f:
        return json.load(f)



def validate_raw_edge_scores(config: OmegaConf, results_root: Path) -> None:
    """Check that a .pt file exists in raw_edge_scores/ for every train example.

    Counts rows in train.csv and verifies that each expected file
    train-{i}-{method}.pt is present.  Raises RuntimeError if any are missing,
    listing the total number missing and the first missing index, with a
    reminder to complete step 02.

    Args:
        config: Full experiment config.
        results_root: Root results directory for this task/model combination.
    """
    raw_dir = results_root / "circuits" / "raw_edge_scores"
    method: str = config.attribution.method

    # Prefer the copy of train.csv stored in results by 02_run_attribution.py;
    # fall back to the original data location from config.
    results_train_csv = results_root / "data" / "train.csv"
    train_csv = (
        results_train_csv
        if results_train_csv.exists()
        else Path(config.paths.output_dir) / "train.csv"
    )
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv}")

    n_train = len(pd.read_csv(train_csv))

    missing = [
        i for i in range(n_train)
        if not (raw_dir / f"train-{i}-{method}.pt").exists()
    ]

    # Allow up to 1% of examples to be missing (skipped due to NaN during IG).
    # These are rare numerical instabilities that cannot be fixed in data generation.
    max_allowed_missing = max(1, int(n_train * 0.01))
    if len(missing) > max_allowed_missing:
        raise RuntimeError(
            f"Validation failed: {len(missing)} of {n_train} raw edge score files are missing "
            f"in {raw_dir}.\n"
            f"First missing index: {missing[0]} "
            f"(expected train-{missing[0]}-{method}.pt).\n"
            "Run 02_run_attribution.py to completion before proceeding."
        )
    elif missing:
        print(
            f"WARNING: {len(missing)} of {n_train} files are missing (NaN-skipped examples): "
            f"{missing}. Proceeding with {n_train - len(missing)} examples."
        )
    else:
        print(f"Validation passed: all {n_train} raw edge score files present in {raw_dir}/")


# Maps attribution method names to output directory slugs.
_METHOD_TO_SLUG = {
    "EAP": "eap",
    "exact": "eact",
}


def _run_attribution(
    model,
    tokenizer,
    csv_path: Path,
    out_dir: Path,
    attr_cfg: OmegaConf,
) -> Path:
    """Run run_localization_dataset and return the saved .pt path.

    Args:
        model: Loaded TransformerLens model.
        tokenizer: Corresponding tokenizer.
        csv_path: Path to the input CSV file.
        out_dir: Directory to save the output .pt graph.
        attr_cfg: Attribution config section from the experiment config.

    Returns:
        Path to the saved .pt file.
    """
    run_localization_dataset(
        model=model,
        tokenizer=tokenizer,
        data_path=str(csv_path),
        method=attr_cfg.method,
        ablation=attr_cfg.ablation,
        ig_steps=attr_cfg.ig_steps,
        batch_size=attr_cfg.batch_size,
        save_dir=str(out_dir),
        level=attr_cfg.level,
    )
    stem = csv_path.stem
    return out_dir / f"{stem}-{attr_cfg.method}.pt"


# ---------------------- Circuit creators ----------------------

def create_dcd_circuits(
    model,
    tokenizer,
    config: OmegaConf,
    results_root: Path,
    best_configs: dict,
    force: bool,
    methods_filter: Optional[set] = None,
) -> None:
    """Create one aggregated attribution circuit per DCD cluster for every method.

    Iterates over all entries in best_configs.json.  For each method, uses its
    best k and the corresponding per-cluster CSVs written by
    03_run_clustering.py, then runs run_localization_dataset() on each cluster
    to produce one .pt graph per cluster.  Outputs land in
    circuits/created/dcd/{method_name}/.

    Args:
        model: Loaded TransformerLens model.
        tokenizer: Corresponding tokenizer.
        config: Full experiment config.
        results_root: Root results directory.
        best_configs: Loaded best_configs.json dict.
        force: Overwrite existing outputs if True.
    """
    attr_cfg = config.attribution

    if methods_filter is not None:
        missing = methods_filter - set(best_configs.keys())
        if missing:
            raise ValueError(
                f"--methods references method(s) not in best_configs.json: "
                f"{sorted(missing)}. Available: {sorted(best_configs.keys())}"
            )

    for method_name, entry in best_configs.items():
        if methods_filter is not None and method_name not in methods_filter:
            continue
        best_k: int = _resolve_best_k(entry)
        result_dir = Path(entry["result_dir"])

        out_dir = results_root / "circuits" / "created" / "dcd" / method_name
        out_dir.mkdir(parents=True, exist_ok=True)

        k_dir = result_dir / f"k_{best_k}"
        cluster_csvs = sorted(k_dir.glob("cluster_*.csv"))
        if not cluster_csvs:
            raise FileNotFoundError(f"No cluster CSVs found in {k_dir}")

        print(f"DCD circuits: method={method_name}, k={best_k}, result_dir={result_dir}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            for csv_path in cluster_csvs:
                expected_out = out_dir / f"{csv_path.stem}-{attr_cfg.method}.pt"
                if expected_out.exists() and not force:
                    print(f"  {expected_out.name} already exists, skipping. Use --force to rerun.")
                    continue

                # Save the cluster dataset alongside the circuit
                shutil.copy(csv_path, out_dir / csv_path.name)

                # Copy to a temp dir so output filename tracks cluster ID, not full path
                tmp_csv = Path(tmp_dir) / csv_path.name
                shutil.copy(csv_path, tmp_csv)

                print(f"  Running attribution for {csv_path.name}")
                _run_attribution(model, tokenizer, tmp_csv, out_dir, attr_cfg)

        print(f"  DCD circuits for {method_name} saved to {out_dir}/")


def create_representative_circuits(
    config: OmegaConf,
    results_root: Path,
    best_configs: dict,
    force: bool,
) -> None:
    """Copy the medoid example's .pt file for each cluster as a representative circuit.

    Reads representative indices from results.json for the best method / best k,
    then locates the corresponding per-example .pt file in raw_edge_scores/ and
    copies it to circuits/created/representative/.

    Args:
        config: Full experiment config.
        results_root: Root results directory.
        best_configs: Loaded best_configs.json dict.
        force: Overwrite existing outputs if True.
    """
    if "kmeans-svd" not in best_configs:
        raise KeyError("'kmeans-svd' entry not found in best_configs.json")
    method_name = "kmeans-svd"
    entry = best_configs[method_name]
    best_k: int = _resolve_best_k(entry)
    result_dir = Path(entry["result_dir"])

    out_dir = results_root / "circuits" / "created" / "representative"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = results_root / "circuits" / "raw_edge_scores"
    attr_cfg = config.attribution

    results_json_path = result_dir / "results.json"
    with open(results_json_path) as f:
        results_data = json.load(f)

    k_key = f"{best_k}-clusters"
    if k_key not in results_data:
        raise KeyError(f"Key '{k_key}' not found in {results_json_path}")

    representatives: Dict[str, int] = results_data[k_key]["representatives"]
    print(f"Representative circuits: method={method_name}, k={best_k}")

    for cluster_id, example_idx in representatives.items():
        dst_pt = out_dir / f"cluster_{cluster_id}_rep{example_idx}-{attr_cfg.method}.pt"
        if dst_pt.exists() and not force:
            print(f"  {dst_pt.name} already exists, skipping.")
            continue

        src_pt = raw_dir / f"train-{example_idx}-{attr_cfg.method}.pt"
        if not src_pt.exists():
            raise FileNotFoundError(
                f"Representative .pt file not found: {src_pt}. "
                "Run 02_run_attribution.py to completion before proceeding."
            )
        shutil.copy(src_pt, dst_pt)
        print(f"  Copied cluster {cluster_id} representative (idx={example_idx}) → {dst_pt.name}")

    print(f"Representative circuits saved to {out_dir}/")


def create_random_k_split_circuits(
    model,
    tokenizer,
    config: OmegaConf,
    results_root: Path,
    best_configs: dict,
    force: bool,
) -> None:
    """Randomly divide train.csv into k groups and create one circuit per group.

    Uses the same k as the overall best DCD method.  Groups are formed by a
    random permutation of example indices split into k equal-sized chunks.

    Args:
        model: Loaded TransformerLens model.
        tokenizer: Corresponding tokenizer.
        config: Full experiment config.
        results_root: Root results directory.
        best_configs: Loaded best_configs.json dict.
        force: Overwrite existing outputs if True.
    """
    if "kmeans-svd" not in best_configs:
        raise KeyError("'kmeans-svd' entry not found in best_configs.json")
    best_k: int = _resolve_best_k(best_configs["kmeans-svd"])

    out_dir = results_root / "circuits" / "created" / "random_k_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    attr_cfg = config.attribution
    train_csv = Path(config.paths.output_dir) / "train.csv"
    train_data = pd.read_csv(train_csv)
    n = len(train_data)

    # Only sample from indices that have a stage-02 per-example .pt file — i.e.
    # examples that did not produce NaN during individual EAP-IG attribution.
    # Including a known-NaN example in a split re-triggers the same failure and
    # kills the batched run.
    raw_dir = results_root / "circuits" / "raw_edge_scores"
    valid_indices = np.array(
        [i for i in range(n) if (raw_dir / f"train-{i}-{attr_cfg.method}.pt").exists()],
        dtype=int,
    )
    n_skipped = n - len(valid_indices)
    if n_skipped:
        skipped = sorted(set(range(n)) - set(valid_indices.tolist()))
        print(f"  Excluding {n_skipped} NaN-skipped indices from random splits: {skipped}")

    random_seed: int = OmegaConf.select(config, "circuit_creation.random_seed", default=42)
    rng = np.random.default_rng(random_seed)
    shuffled = valid_indices[rng.permutation(len(valid_indices))]
    split_indices = np.array_split(shuffled, best_k)

    print(f"Random k-split circuits: k={best_k}, n={len(valid_indices)}, seed={random_seed}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split_id, group_idx in enumerate(split_indices):
            expected_out = out_dir / f"split_{split_id}-{attr_cfg.method}.pt"
            if expected_out.exists() and not force:
                print(f"  {expected_out.name} already exists, skipping. Use --force to rerun.")
                continue

            # Save the split dataset alongside the circuit
            split_csv = out_dir / f"split_{split_id}.csv"
            train_data.iloc[group_idx].to_csv(split_csv, index=False)

            tmp_csv = Path(tmp_dir) / f"split_{split_id}.csv"
            shutil.copy(split_csv, tmp_csv)

            print(f"  Running attribution for split {split_id} ({len(group_idx)} examples)")
            try:
                _run_attribution(model, tokenizer, tmp_csv, out_dir, attr_cfg)
            except ValueError as e:
                # Some examples trigger NaN only when batched/sequenced through a
                # shared accumulator even if their individual stage-02 run was
                # fine. Fall back to averaging the pre-computed per-example
                # edge scores for this split — mathematically equivalent to
                # batched EAP-IG aggregation and NaN-free.
                print(
                    f"  WARNING: batched attribution for split {split_id} raised "
                    f"{type(e).__name__}: {e}. Falling back to averaging per-example "
                    f"scores from {raw_dir}/"
                )
                _average_raw_scores_to_pt(
                    raw_dir=raw_dir,
                    method=attr_cfg.method,
                    group_idx=group_idx,
                    out_path=expected_out,
                )

    print(f"Random k-split circuits saved to {out_dir}/")


def _average_raw_scores_to_pt(
    raw_dir: Path,
    method: str,
    group_idx: np.ndarray,
    out_path: Path,
) -> None:
    """Aggregate stage-02 per-example edge scores into a single group circuit.

    Loads each train-{i}-{method}.pt in group_idx, averages their edges_scores
    tensors, and writes a new .pt file at out_path using the first loaded file
    as a structural template. Used as a NaN-safe fallback when batched
    attribution on a random split fails.

    Args:
        raw_dir: Directory holding per-example stage-02 .pt files.
        method: Attribution method string (matches filename suffix).
        group_idx: Original train indices to aggregate over.
        out_path: Destination .pt path for the averaged group circuit.
    """
    paths = [raw_dir / f"train-{int(i)}-{method}.pt" for i in group_idx]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise RuntimeError(
            f"No per-example .pt files available in {raw_dir} for the split — "
            "cannot fall back to averaging."
        )
    template = torch.load(paths[0], map_location="cpu", weights_only=False)
    agg = torch.zeros_like(template["edges_scores"])
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        agg = agg + d["edges_scores"]
    agg = agg / len(paths)
    template["edges_scores"] = agg
    torch.save(template, out_path)
    print(f"  Fallback: averaged {len(paths)} per-example scores → {out_path.name}")


def create_baseline_circuits(
    model,
    tokenizer,
    config: OmegaConf,
    results_root: Path,
    force: bool,
) -> None:
    """Create one circuit per baseline attribution method (EAP, E-Act).

    Reads circuit_creation.baseline_methods from config and runs attribution
    on the full train.csv for each method.  Circuits are saved under
    circuits/created/{slug}/train-{method}.pt where slug comes from
    _METHOD_TO_SLUG.  E-Act (exact) respects circuit_creation.eact_max_examples.

    Args:
        model: Loaded TransformerLens model.
        tokenizer: Corresponding tokenizer.
        config: Full experiment config.
        results_root: Root results directory.
        force: Overwrite existing outputs if True.
    """
    baseline_methods = OmegaConf.select(config, "circuit_creation.baseline_methods", default=[])
    if not baseline_methods:
        print("No baseline_methods configured in circuit_creation, skipping.")
        return

    attr_cfg = config.attribution
    eact_max_examples: Optional[int] = OmegaConf.select(
        config, "circuit_creation.eact_max_examples", default=None
    )
    eact_batch_size: int = OmegaConf.select(
        config, "circuit_creation.eact_batch_size", default=attr_cfg.batch_size
    )
    train_csv = Path(config.paths.output_dir) / "train.csv"

    for method in baseline_methods:
        slug = _METHOD_TO_SLUG.get(method, method.lower())
        out_dir = results_root / "circuits" / "created" / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        expected_out = out_dir / f"train-{method}.pt"
        if expected_out.exists() and not force:
            print(f"  [{slug}] {expected_out.name} already exists, skipping. Use --force to rerun.")
            continue

        max_examples = eact_max_examples if method == "exact" else None
        batch_size = eact_batch_size if method == "exact" else attr_cfg.batch_size
        print(f"  [{slug}] Running attribution (method={method}, batch_size={batch_size}, max_examples={max_examples})")
        run_localization_dataset(
            model=model,
            tokenizer=tokenizer,
            data_path=str(train_csv),
            method=method,
            ablation=attr_cfg.ablation,
            ig_steps=attr_cfg.ig_steps,
            batch_size=batch_size,
            save_dir=str(out_dir),
            level=attr_cfg.level,
            max_examples=max_examples,
        )
        print(f"  [{slug}] Circuit saved to {out_dir}/")


def create_single_circuit(
    model,
    tokenizer,
    config: OmegaConf,
    results_root: Path,
    force: bool,
) -> None:
    """Run attribution on the full train.csv to produce one baseline circuit.

    Args:
        model: Loaded TransformerLens model.
        tokenizer: Corresponding tokenizer.
        config: Full experiment config.
        results_root: Root results directory.
        force: Overwrite existing output if True.
    """
    out_dir = results_root / "circuits" / "created" / "single"
    out_dir.mkdir(parents=True, exist_ok=True)

    attr_cfg = config.attribution
    train_csv = Path(config.paths.output_dir) / "train.csv"

    expected_out = out_dir / f"train-{attr_cfg.method}.pt"
    if expected_out.exists() and not force:
        print(f"Single circuit already exists at {expected_out}. Use --force to rerun.")
        return

    # Filter stage-02 NaN-skipped indices before batched attribution — same
    # reason as create_random_k_split_circuits: those examples deterministically
    # re-trigger NaN and kill the whole run.
    raw_dir = results_root / "circuits" / "raw_edge_scores"
    train_data = pd.read_csv(train_csv)
    n = len(train_data)
    valid_indices = np.array(
        [i for i in range(n) if (raw_dir / f"train-{i}-{attr_cfg.method}.pt").exists()],
        dtype=int,
    )
    n_skipped = n - len(valid_indices)
    if n_skipped:
        skipped = sorted(set(range(n)) - set(valid_indices.tolist()))
        print(f"  Excluding {n_skipped} NaN-skipped indices from single circuit: {skipped}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        filtered_csv = Path(tmp_dir) / "train.csv"
        train_data.iloc[valid_indices].to_csv(filtered_csv, index=False)
        print(f"Single circuit: running attribution on filtered train.csv (n={len(valid_indices)})")
        try:
            _run_attribution(model, tokenizer, filtered_csv, out_dir, attr_cfg)
        except ValueError as e:
            print(
                f"  WARNING: batched attribution for single circuit raised "
                f"{type(e).__name__}: {e}. Falling back to averaging per-example "
                f"scores from {raw_dir}/"
            )
            _average_raw_scores_to_pt(
                raw_dir=raw_dir,
                method=attr_cfg.method,
                group_idx=valid_indices,
                out_path=expected_out,
            )
    print(f"Single circuit saved to {out_dir}/")


def create_random_circuit(
    config: OmegaConf,
    results_root: Path,
    force: bool,
) -> None:
    """Generate a random circuit by sampling uniform edge scores.

    Loads train-0-{method}.pt from raw_edge_scores/ to determine the number of
    edges, then draws scores from Uniform(-1, 1) and saves as random_circuit.npy.

    Args:
        config: Full experiment config.
        results_root: Root results directory.
        force: Overwrite existing output if True.
    """
    out_dir = results_root / "circuits" / "created" / "random"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "random_circuit.npy"
    if out_path.exists() and not force:
        print(f"Random circuit already exists at {out_path}. Use --force to rerun.")
        return

    raw_dir = results_root / "circuits" / "raw_edge_scores"
    method: str = config.attribution.method
    ref_pt = raw_dir / f"train-0-{method}.pt"
    if not ref_pt.exists():
        raise FileNotFoundError(
            f"Reference .pt file not found: {ref_pt}. "
            "Run 02_run_attribution.py to completion before proceeding."
        )

    ref_scores = extract_edge_scores(str(ref_pt))
    n_edges = ref_scores.shape[0]

    rng = np.random.default_rng(42)
    random_scores = rng.uniform(-1.0, 1.0, size=(n_edges,))
    np.save(out_path, random_scores)
    print(f"Random circuit saved to {out_path} (n_edges={n_edges})")


# ---------------------- Entry point ----------------------

def main() -> None:
    """Entry point for DCD stage 4: create circuits from clustering results."""
    parser = argparse.ArgumentParser(
        description="DCD stage 4: create circuits from clustering and attribution.",
    )
    parser.add_argument("--config", required=True, help="Path to model-specific YAML config")
    parser.add_argument(
        "--circuit-type",
        default="all",
        choices=["all", "dcd", "representative", "random_k_split", "single", "random", "baseline"],
        help="Which circuit type(s) to create (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated list of DCD method names to create circuits for "
             "(--circuit-type dcd only). Default 'all' runs every method in "
             "best_configs.json.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. circuit_creation.random_seed=0",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    results_root = _results_root(config)

    shutil.copy(args.config, results_root)
    print(f"Copied config to {results_root}/")

    validate_raw_edge_scores(config, results_root)

    run_all = args.circuit_type == "all"
    needs_best_configs = run_all or args.circuit_type in {"dcd", "representative", "random_k_split"}
    needs_model = run_all or args.circuit_type in {"dcd", "random_k_split", "single", "baseline"}

    best_configs: Optional[dict] = None
    if needs_best_configs:
        best_configs = _load_best_configs(results_root)

    model = tokenizer = None
    if needs_model:
        model, tokenizer = load_model(config.model.name, config.model.cache_dir)
        print(f"Loaded model {config.model.name}")

    methods_filter: Optional[set] = None
    if args.methods and args.methods != "all":
        methods_filter = {m.strip() for m in args.methods.split(",") if m.strip()}

    if run_all or args.circuit_type == "dcd":
        create_dcd_circuits(
            model, tokenizer, config, results_root, best_configs, args.force,
            methods_filter=methods_filter,
        )

    if run_all or args.circuit_type == "representative":
        create_representative_circuits(
            config, results_root, best_configs, args.force
        )

    if run_all or args.circuit_type == "random_k_split":
        create_random_k_split_circuits(
            model, tokenizer, config, results_root, best_configs, args.force
        )

    if run_all or args.circuit_type == "single":
        create_single_circuit(model, tokenizer, config, results_root, args.force)

    if run_all or args.circuit_type == "random":
        create_random_circuit(config, results_root, args.force)

    if run_all or args.circuit_type == "baseline":
        create_baseline_circuits(model, tokenizer, config, results_root, args.force)

    print("\nCircuit creation complete.")


if __name__ == "__main__":
    main()
