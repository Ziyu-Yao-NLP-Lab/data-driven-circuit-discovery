"""DCD stage 5: evaluate faithfulness of all created circuit types.

Usage:
    python experiments/dcd/05_evaluate_faithfulness.py --config configs/dcd/all-tasks/gpt2.yaml
    python experiments/dcd/05_evaluate_faithfulness.py --config configs/dcd/all-tasks/gpt2.yaml \
        --circuit-type dcd --force
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from eap.graph import Graph
from src.circuit_discovery.faithfulness import (
    compute_cpr_cmd,
    compute_per_example_faithfulness,
    run_faithfulness,
    run_faithfulness_multi_circuit,
)
from src.utils import general_utils as gu
from src.utils import metrics
from src.utils.graph_utils import extract_edge_scores


# ---------------------- Helpers ----------------------


def _results_root(config: OmegaConf) -> Path:
    task_name: str = config.data.task_name
    model_dir: str = Path(config.paths.output_dir).name
    return Path(f"results/dcd/{task_name}/{model_dir}")


def _eval_dir(results_root: Path) -> Path:
    return results_root / "evaluation"


def _load_pt_paths(directory: Path, pattern: str) -> List[Path]:
    """Return sorted .pt files matching pattern under directory."""
    return sorted(directory.glob(pattern))


def _per_example_eval_dir(results_root: Path) -> Path:
    """Per-example faithfulness matrices live alongside other analysis outputs."""
    return results_root / "analysis" / "evaluation"


def _save_per_example_matrix(
    out_path: Path,
    model,
    metric,
    circuit_paths: List[Path],
    data_path: str,
    force: bool,
    label: str,
) -> None:
    """Compute and save a (k, n_sizes, n_test) per-example faithfulness npz.

    Skips re-computation if the file already exists and ``force`` is False.

    Args:
        out_path: Destination .npz path.
        model: Loaded TransformerLens model.
        metric: Metric closure (must accept mean=False).
        circuit_paths: Circuit .pt paths to evaluate.
        data_path: Test CSV path.
        force: Recompute even if out_path already exists.
        label: Short label used in log lines.
    """
    if out_path.exists() and not force:
        print(f"  [{label}] per-example matrix already at {out_path}, skipping.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{label}] computing per-example faithfulness ({len(circuit_paths)} circuits) ...")
    result = compute_per_example_faithfulness(
        model=model,
        circuit_paths=[str(p) for p in circuit_paths],
        data_path=data_path,
        metric=metric,
    )
    np.savez(
        out_path,
        faith=result["faith"],
        circuit_score=result["circuit_score"],
        sizes=result["sizes"],
        baseline_per_example=result["baseline_per_example"],
        corrupted_per_example=result["corrupted_per_example"],
        circuit_files=np.array([Path(p).name for p in circuit_paths]),
    )
    print(f"  [{label}] saved per-example matrix to {out_path}  shape={result['faith'].shape}")


def _random_npy_to_pt(
    npy_path: Path,
    ref_pt_path: Path,
    tmp_dir: str,
) -> Path:
    """Convert random_circuit.npy into a temporary .pt Graph file.

    Loads the reference graph to get graph topology, assigns the random
    edge scores (in the same order extract_edge_scores returns them),
    and saves to a temp .pt file.

    Args:
        npy_path: Path to random_circuit.npy.
        ref_pt_path: Path to a reference .pt graph (same model/topology).
        tmp_dir: Directory to write the temporary .pt file.

    Returns:
        Path to the generated temporary .pt file.
    """
    random_scores = np.load(npy_path)
    graph = Graph.from_pt(str(ref_pt_path))
    edge_names = list(graph.edges.keys())

    if len(edge_names) != len(random_scores):
        raise ValueError(
            f"Edge count mismatch: graph has {len(edge_names)} edges "
            f"but random_circuit.npy has {len(random_scores)} scores."
        )

    for i, name in enumerate(edge_names):
        graph.edges[name].score = float(random_scores[i])

    out_path = Path(tmp_dir) / "random_circuit.pt"
    graph.to_pt(str(out_path))
    return out_path


# ---------------------- Per-type evaluators ----------------------


def evaluate_dcd(
    model,
    metric,
    results_root: Path,
    data_path: str,
    force: bool,
    methods_filter: Optional[set] = None,
) -> Tuple[Dict, Dict]:
    """Evaluate each DCD clustering method's circuits.

    Returns:
        faith_results: method_name → faithfulness result dict
        cpr_cmd: method_name → {cpr, cmd}
    """
    dcd_dir = results_root / "circuits" / "created" / "dcd"
    if not dcd_dir.exists():
        print(f"  [dcd] No directory found at {dcd_dir}, skipping.")
        return {}, {}

    method_dirs = sorted(dcd_dir.iterdir())
    if methods_filter is not None:
        available = {d.name for d in method_dirs if d.is_dir()}
        missing = methods_filter - available
        if missing:
            raise ValueError(
                f"--methods references method(s) with no circuits directory: "
                f"{sorted(missing)}. Available under {dcd_dir}: {sorted(available)}"
            )
    faith_results: Dict = {}
    cpr_cmd: Dict = {}

    for method_dir in method_dirs:
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name
        if methods_filter is not None and method_name not in methods_filter:
            continue
        circuit_paths = _load_pt_paths(method_dir, "cluster_*.pt")
        if not circuit_paths:
            print(f"  [dcd/{method_name}] No cluster_*.pt files found, skipping.")
            continue

        print(f"\n  [dcd/{method_name}] {len(circuit_paths)} circuits → {data_path}")
        result = run_faithfulness_multi_circuit(
            model=model,
            circuit_paths=[str(p) for p in circuit_paths],
            data_path=data_path,
            metric=metric,
        )
        faith_results[method_name] = result

        cpr, cmd = compute_cpr_cmd(result["faithfulness"])
        cpr_cmd[method_name] = {"cpr": cpr, "cmd": cmd}
        print(f"    CPR={cpr:.4f}  CMD={cmd:.4f}")

        per_ex_path = _per_example_eval_dir(results_root) / "dcd" / f"{method_name}.npz"
        _save_per_example_matrix(
            out_path=per_ex_path,
            model=model,
            metric=metric,
            circuit_paths=circuit_paths,
            data_path=data_path,
            force=force,
            label=f"dcd/{method_name}",
        )

    return faith_results, cpr_cmd


def evaluate_representative(
    model,
    metric,
    results_root: Path,
    data_path: str,
    force: bool,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    rep_dir = results_root / "circuits" / "created" / "representative"
    if not rep_dir.exists():
        print(f"  [representative] No directory found at {rep_dir}, skipping.")
        return None, None

    circuit_paths = _load_pt_paths(rep_dir, "cluster_*_rep*.pt")
    if not circuit_paths:
        print(f"  [representative] No cluster_*_rep*.pt files found, skipping.")
        return None, None

    print(f"\n  [representative] {len(circuit_paths)} circuits → {data_path}")
    result = run_faithfulness_multi_circuit(
        model=model,
        circuit_paths=[str(p) for p in circuit_paths],
        data_path=data_path,
        metric=metric,
    )
    cpr, cmd = compute_cpr_cmd(result["faithfulness"])
    print(f"    CPR={cpr:.4f}  CMD={cmd:.4f}")
    return result, {"cpr": cpr, "cmd": cmd}


def evaluate_random_k_split(
    model,
    metric,
    results_root: Path,
    data_path: str,
    force: bool,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    rks_dir = results_root / "circuits" / "created" / "random_k_split"
    if not rks_dir.exists():
        print(f"  [random_k_split] No directory found at {rks_dir}, skipping.")
        return None, None

    circuit_paths = _load_pt_paths(rks_dir, "split_*.pt")
    if not circuit_paths:
        print(f"  [random_k_split] No split_*.pt files found, skipping.")
        return None, None

    print(f"\n  [random_k_split] {len(circuit_paths)} circuits → {data_path}")
    result = run_faithfulness_multi_circuit(
        model=model,
        circuit_paths=[str(p) for p in circuit_paths],
        data_path=data_path,
        metric=metric,
    )
    cpr, cmd = compute_cpr_cmd(result["faithfulness"])
    print(f"    CPR={cpr:.4f}  CMD={cmd:.4f}")

    per_ex_path = _per_example_eval_dir(results_root) / "random_k_split.npz"
    _save_per_example_matrix(
        out_path=per_ex_path,
        model=model,
        metric=metric,
        circuit_paths=circuit_paths,
        data_path=data_path,
        force=force,
        label="random_k_split",
    )

    return result, {"cpr": cpr, "cmd": cmd}


def evaluate_single(
    model,
    metric,
    results_root: Path,
    data_path: str,
    force: bool,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    single_dir = results_root / "circuits" / "created" / "single"
    if not single_dir.exists():
        print(f"  [single] No directory found at {single_dir}, skipping.")
        return None, None

    pt_files = _load_pt_paths(single_dir, "*.pt")
    if not pt_files:
        print(f"  [single] No .pt file found, skipping.")
        return None, None
    if len(pt_files) > 1:
        print(f"  [single] Warning: found {len(pt_files)} .pt files, using {pt_files[0].name}.")

    circuit_path = pt_files[0]
    print(f"\n  [single] {circuit_path.name} → {data_path}")
    result = run_faithfulness(
        model=model,
        circuit_path=str(circuit_path),
        data_path=data_path,
        metric=metric,
    )
    cpr, cmd = compute_cpr_cmd(result["faithfulness"])
    print(f"    CPR={cpr:.4f}  CMD={cmd:.4f}")
    return result, {"cpr": cpr, "cmd": cmd}


def evaluate_baseline_circuits(
    model,
    metric,
    config: OmegaConf,
    results_root: Path,
    data_path: str,
    force: bool,
) -> Tuple[Dict, Dict]:
    """Evaluate EAP and E-Act single-circuit baselines.

    Reads circuit_creation.baseline_methods from config, locates each circuit
    under circuits/created/{slug}/train-{method}.pt, and evaluates faithfulness.

    Args:
        model: Loaded TransformerLens model.
        metric: Attribution metric callable.
        config: Full experiment config.
        results_root: Root results directory.
        data_path: Path to the test CSV.
        force: Rerun even if results already cached.

    Returns:
        faith_results: slug → faithfulness result dict
        cpr_cmd: slug → {cpr, cmd}
    """
    _METHOD_TO_SLUG = {"EAP": "eap", "exact": "eact"}

    baseline_methods = OmegaConf.select(config, "circuit_creation.baseline_methods", default=[])
    if not baseline_methods:
        print("  [baseline] No baseline_methods configured, skipping.")
        return {}, {}

    faith_results: Dict = {}
    cpr_cmd: Dict = {}

    for method in baseline_methods:
        slug = _METHOD_TO_SLUG.get(method, method.lower())
        circuit_dir = results_root / "circuits" / "created" / slug
        if not circuit_dir.exists():
            print(f"  [baseline/{slug}] Directory not found at {circuit_dir}, skipping.")
            continue

        pt_files = sorted(circuit_dir.glob("*.pt"))
        if not pt_files:
            print(f"  [baseline/{slug}] No .pt file found, skipping.")
            continue
        if len(pt_files) > 1:
            print(f"  [baseline/{slug}] Warning: found {len(pt_files)} .pt files, using {pt_files[0].name}.")

        circuit_path = pt_files[0]
        print(f"\n  [baseline/{slug}] {circuit_path.name} → {data_path}")
        result = run_faithfulness(
            model=model,
            circuit_path=str(circuit_path),
            data_path=data_path,
            metric=metric,
        )
        cpr, cmd = compute_cpr_cmd(result["faithfulness"])
        faith_results[slug] = result
        cpr_cmd[slug] = {"cpr": cpr, "cmd": cmd}
        print(f"    CPR={cpr:.4f}  CMD={cmd:.4f}")

    return faith_results, cpr_cmd


def evaluate_random(
    model,
    metric,
    config: OmegaConf,
    results_root: Path,
    data_path: str,
    force: bool,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    random_dir = results_root / "circuits" / "created" / "random"
    npy_path = random_dir / "random_circuit.npy"
    if not npy_path.exists():
        print(f"  [random] random_circuit.npy not found at {npy_path}, skipping.")
        return None, None

    # Find reference .pt to clone graph topology
    raw_dir = results_root / "circuits" / "raw_edge_scores"
    method: str = config.attribution.method
    ref_pt = raw_dir / f"train-0-{method}.pt"
    if not ref_pt.exists():
        raise FileNotFoundError(
            f"Reference .pt not found: {ref_pt}. "
            "Run 02_run_attribution.py before evaluating the random circuit."
        )

    print(f"\n  [random] converting npy → temp .pt, then evaluating → {data_path}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pt = _random_npy_to_pt(npy_path, ref_pt, tmp_dir)
        result = run_faithfulness(
            model=model,
            circuit_path=str(tmp_pt),
            data_path=data_path,
            metric=metric,
        )

    cpr, cmd = compute_cpr_cmd(result["faithfulness"])
    print(f"    CPR={cpr:.4f}  CMD={cmd:.4f}")
    return result, {"cpr": cpr, "cmd": cmd}


# ---------------------- Summary table ----------------------

_DISPLAY_SIZES = ["0.001", "0.01", "0.05", "0.1", "0.5", "1"]


def _print_summary(faith_results: Dict, cpr_cmd_results: Dict) -> None:
    """Print a comparison table of faithfulness across circuit types."""
    print("\n" + "=" * 90)
    print("FAITHFULNESS SUMMARY")
    print("=" * 90)

    header = f"{'Circuit type':<35}" + "".join(f"{s:>9}" for s in _DISPLAY_SIZES) + f"{'CPR':>9}{'CMD':>9}"
    print(header)
    print("-" * 90)

    def _row(label: str, faith_dict: Dict, cpr: float, cmd: float) -> None:
        vals = "".join(
            f"{faith_dict.get(s, float('nan')):>9.4f}" for s in _DISPLAY_SIZES
        )
        print(f"{label:<35}{vals}{cpr:>9.4f}{cmd:>9.4f}")

    # DCD methods
    dcd_faith = faith_results.get("dcd", {})
    dcd_cpr_cmd = cpr_cmd_results.get("dcd", {})
    for method_name in sorted(dcd_faith.keys()):
        result = dcd_faith[method_name]
        cc = dcd_cpr_cmd.get(method_name, {})
        _row(
            f"dcd/{method_name}",
            result.get("faithfulness", {}),
            cc.get("cpr", float("nan")),
            cc.get("cmd", float("nan")),
        )

    # Other single-entry types
    for key in ("representative", "random_k_split", "single", "random"):
        if key not in faith_results:
            continue
        result = faith_results[key]
        cc = cpr_cmd_results.get(key, {})
        _row(
            key,
            result.get("faithfulness", {}),
            cc.get("cpr", float("nan")),
            cc.get("cmd", float("nan")),
        )

    # Baseline circuits (EAP, E-Act)
    baseline_faith = faith_results.get("baseline", {})
    baseline_cpr_cmd = cpr_cmd_results.get("baseline", {})
    for slug in sorted(baseline_faith.keys()):
        result = baseline_faith[slug]
        cc = baseline_cpr_cmd.get(slug, {})
        _row(
            f"baseline/{slug}",
            result.get("faithfulness", {}),
            cc.get("cpr", float("nan")),
            cc.get("cmd", float("nan")),
        )

    print("=" * 90)


# ---------------------- Entry point ----------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DCD stage 5: evaluate faithfulness of all created circuit types."
    )
    parser.add_argument("--config", required=True, help="Path to model-specific YAML config")
    parser.add_argument(
        "--circuit-type",
        default="all",
        choices=["all", "dcd", "representative", "random_k_split", "single", "random", "baseline"],
        help="Which circuit type(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Rerun evaluation even if results already exist",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated list of DCD method names to evaluate "
             "(--circuit-type dcd only). Default 'all' evaluates every "
             "method directory under circuits/created/dcd/.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    results_root = _results_root(config)
    eval_dir = _eval_dir(results_root)
    eval_dir.mkdir(parents=True, exist_ok=True)

    faith_out = eval_dir / "faithfulness_results.json"
    cpr_cmd_out = eval_dir / "cpr_cmd_results.json"

    if (
        faith_out.exists()
        and cpr_cmd_out.exists()
        and not args.force
        and args.circuit_type == "all"
        and args.methods == "all"
    ):
        print(
            f"Results already exist in {eval_dir}. "
            "Pass --force to rerun."
        )
        with open(faith_out) as f:
            faith_results = json.load(f)
        with open(cpr_cmd_out) as f:
            cpr_cmd_results = json.load(f)
        _print_summary(faith_results, cpr_cmd_results)
        return

    # Copy config for reproducibility
    shutil.copy(args.config, eval_dir / Path(args.config).name)
    print(f"Copied config to {eval_dir}/")

    # Resolve test data path
    data_path_cfg = OmegaConf.select(config, "evaluation.test_data")
    if data_path_cfg:
        data_path = str(data_path_cfg)
    else:
        data_path = str(Path(config.paths.output_dir) / "test.csv")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Test data not found: {data_path}")
    print(f"Test data: {data_path}")

    # Load model once for all evaluations
    print(f"Loading model {config.model.name} ...")
    model, tokenizer = gu.load_model(config.model.name, config.model.cache_dir)
    metric = metrics.get_logit_diff(model.tokenizer)
    print("Model loaded.")

    run_all = args.circuit_type == "all"
    methods_filter: Optional[set] = None
    if args.methods and args.methods != "all":
        methods_filter = {m.strip() for m in args.methods.split(",") if m.strip()}

    # When re-running a subset (--circuit-type != all or --methods filter),
    # start from existing files so we don't clobber circuit types / methods
    # that aren't being re-evaluated this run.
    faith_results: Dict = {}
    cpr_cmd_results: Dict = {}
    if faith_out.exists() and cpr_cmd_out.exists():
        with open(faith_out) as f:
            faith_results = json.load(f)
        with open(cpr_cmd_out) as f:
            cpr_cmd_results = json.load(f)
        print(f"Loaded existing results from {eval_dir} — new entries will be merged in.")

    # --- DCD methods ---
    if run_all or args.circuit_type == "dcd":
        print("\n=== Evaluating DCD circuits ===")
        dcd_faith, dcd_cpr_cmd = evaluate_dcd(
            model, metric, results_root, data_path, args.force,
            methods_filter=methods_filter,
        )
        if dcd_faith:
            # Merge per-method so existing DCD entries outside the filter survive
            existing_dcd = faith_results.get("dcd", {}) if isinstance(faith_results.get("dcd"), dict) else {}
            existing_dcd_cpr = cpr_cmd_results.get("dcd", {}) if isinstance(cpr_cmd_results.get("dcd"), dict) else {}
            faith_results["dcd"] = {**existing_dcd, **dcd_faith}
            cpr_cmd_results["dcd"] = {**existing_dcd_cpr, **dcd_cpr_cmd}

    # --- Representative ---
    if run_all or args.circuit_type == "representative":
        print("\n=== Evaluating representative circuits ===")
        rep_faith, rep_cpr = evaluate_representative(
            model, metric, results_root, data_path, args.force
        )
        if rep_faith is not None:
            faith_results["representative"] = rep_faith
            cpr_cmd_results["representative"] = rep_cpr

    # --- Random k-split ---
    if run_all or args.circuit_type == "random_k_split":
        print("\n=== Evaluating random_k_split circuits ===")
        rks_faith, rks_cpr = evaluate_random_k_split(
            model, metric, results_root, data_path, args.force
        )
        if rks_faith is not None:
            faith_results["random_k_split"] = rks_faith
            cpr_cmd_results["random_k_split"] = rks_cpr

    # --- Single ---
    if run_all or args.circuit_type == "single":
        print("\n=== Evaluating single circuit ===")
        sin_faith, sin_cpr = evaluate_single(
            model, metric, results_root, data_path, args.force
        )
        if sin_faith is not None:
            faith_results["single"] = sin_faith
            cpr_cmd_results["single"] = sin_cpr

    # --- Random ---
    if run_all or args.circuit_type == "random":
        print("\n=== Evaluating random circuit ===")
        rnd_faith, rnd_cpr = evaluate_random(
            model, metric, config, results_root, data_path, args.force
        )
        if rnd_faith is not None:
            faith_results["random"] = rnd_faith
            cpr_cmd_results["random"] = rnd_cpr

    # --- Baseline (EAP, E-Act) ---
    if run_all or args.circuit_type == "baseline":
        print("\n=== Evaluating baseline circuits (EAP, E-Act) ===")
        bl_faith, bl_cpr = evaluate_baseline_circuits(
            model, metric, config, results_root, data_path, args.force
        )
        if bl_faith:
            existing_bl = faith_results.get("baseline", {}) if isinstance(faith_results.get("baseline"), dict) else {}
            existing_bl_cpr = cpr_cmd_results.get("baseline", {}) if isinstance(cpr_cmd_results.get("baseline"), dict) else {}
            faith_results["baseline"] = {**existing_bl, **bl_faith}
            cpr_cmd_results["baseline"] = {**existing_bl_cpr, **bl_cpr}

    # Surface the running NaN counter from the metric closure. We replace
    # per-example NaN with 0 inside metrics.get_logit_diff (numerical
    # instability in heavily-pruned circuits with bf16/fp16), and want a
    # paper-trail of how often that fires.
    nan_count = int(getattr(metric, "nan_count", 0))
    eval_count = int(getattr(metric, "eval_count", 0))
    nan_fraction = (nan_count / eval_count) if eval_count else 0.0
    nan_summary = {
        "nan_examples_replaced": nan_count,
        "total_examples_evaluated": eval_count,
        "nan_fraction": nan_fraction,
    }
    faith_results["_metric_nan_summary"] = nan_summary
    cpr_cmd_results["_metric_nan_summary"] = nan_summary
    print(
        f"\nNaN substitution summary: {nan_count}/{eval_count} examples "
        f"({nan_fraction:.2%}) had NaN logit_diff and were replaced with 0."
    )

    # Save results
    gu.save_json_file(faith_results, str(faith_out))
    gu.save_json_file(cpr_cmd_results, str(cpr_cmd_out))
    print(f"\nFaithfulness results saved to: {faith_out}")
    print(f"CPR/CMD results saved to:      {cpr_cmd_out}")

    _print_summary(faith_results, cpr_cmd_results)


if __name__ == "__main__":
    main()
