"""Evaluate dataset-specific circuits across mixing ratios for Experiment 2.

For each (method, mixing_ratio) discovered by 02_run_circuit_discovery.py:
  1. Faithfulness on the mixed test set at that ratio.
  2. Faithfulness on the pure entity-binding test set (ratio = 1.0).
  3. Faithfulness on the pure arithmetic test set (ratio = 0.0).
  4. Circuit size (number of edges retained at gamma=0.99 of total absolute
     attribution).

Each faithfulness curve is collapsed into a single CPR / CMD scalar (same
convention as Experiment 1). Outputs land in:
  results/dataset-specific-circuits/{model}/evaluation/{method}/
    faithfulness_results.json
    summary_table.txt
    summary_table.csv

The produced CSV is the source for the two headline plots:
  x = mixing_ratio vs y = faithfulness (three lines: mixed, eb, arith)
  x = mixing_ratio vs y = circuit_size
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

from omegaconf import OmegaConf

from src.circuit_discovery.faithfulness import run_faithfulness, compute_cpr_cmd
from src.dcd.clustering import sparsify_binary
from src.utils import metrics
from src.utils.general_utils import load_model, model2family
from src.utils.graph_utils import build_edge_index, load_scores_fast

GAMMA = 0.99
TEST_SETS = ("mixed", "entity-binding", "arithmetic")


def parse_list(flag_value: Optional[str]) -> Optional[List[str]]:
    if flag_value is None:
        return None
    return [x.strip() for x in flag_value.split(",") if x.strip()]


def _ratio_str(ratio: float) -> str:
    return f"{ratio:.1f}"


def _circuit_size(circuit_path: str, gamma: float) -> int:
    """Number of edges retained when binarizing at gamma cumulative attribution."""
    _, rows, cols = build_edge_index(circuit_path)
    scores = load_scores_fast(circuit_path, rows, cols)
    mask = sparsify_binary(scores, gamma)
    return int(mask.sum())


def _write_summary_tables(
    method: str,
    method_eval_dir: str,
    results: Dict[str, dict],
) -> None:
    """Write the (ratio × test_set) table in CSV and text form.

    Rows are ordered by ratio then by test_set. circuit_size is repeated
    per row — it is a property of the circuit, not the test set, but
    duplication keeps the CSV easy to pivot for plotting.
    """
    rows = []
    for ratio in sorted(results.keys(), key=lambda r: float(r)):
        entry = results[ratio]
        circuit_size = entry["circuit_size"]
        for test_set in TEST_SETS:
            test_entry = entry["faithfulness"].get(test_set)
            if test_entry is None:
                continue
            rows.append(
                {
                    "method": method,
                    "mixing_ratio": ratio,
                    "test_set": test_set,
                    "cpr": test_entry["cpr"],
                    "cmd": test_entry["cmd"],
                    "baseline_score": test_entry["baseline_score"],
                    "corrupted_score": test_entry["corrupted_score"],
                    "circuit_size": circuit_size,
                }
            )

    csv_path = os.path.join(method_eval_dir, "summary_table.csv")
    fieldnames = [
        "method",
        "mixing_ratio",
        "test_set",
        "cpr",
        "cmd",
        "baseline_score",
        "corrupted_score",
        "circuit_size",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = os.path.join(method_eval_dir, "summary_table.txt")
    with open(txt_path, "w") as f:
        f.write(
            f"{'method':<16}{'ratio':<8}{'test_set':<18}"
            f"{'cpr':<10}{'cmd':<10}{'size':<10}\n"
        )
        f.write("-" * 72 + "\n")
        for row in rows:
            f.write(
                f"{row['method']:<16}{row['mixing_ratio']:<8}"
                f"{row['test_set']:<18}"
                f"{row['cpr']:<10.4f}{row['cmd']:<10.4f}"
                f"{row['circuit_size']:<10d}\n"
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
        "--ratios",
        type=str,
        default=None,
        help="Comma-separated mixing ratios to evaluate (subset of config ratios)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluation even if outputs already exist",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model_family = model2family(config.model.name)

    method_filter = parse_list(args.methods)
    all_methods = [m.name for m in config.attribution.methods]
    methods = [m for m in all_methods if method_filter is None or m in method_filter]
    if not methods:
        print("No methods selected — check --methods.")
        return

    all_ratios = [float(r) for r in config.data.mixing_ratios]
    ratio_filter = parse_list(args.ratios)
    if ratio_filter is not None:
        keep = {float(r) for r in ratio_filter}
        all_ratios = [r for r in all_ratios if r in keep]
    if not all_ratios:
        print("No ratios selected — check --ratios.")
        return

    results_root = f"results/dataset-specific-circuits/{model_family}"
    circuits_root = os.path.join(results_root, "circuits")
    eval_root = os.path.join(results_root, "evaluation")
    os.makedirs(eval_root, exist_ok=True)
    shutil.copy(args.config, results_root)

    data_root = os.path.join(config.paths.output_dir, model_family)

    # Test sets: mixed (ratio-specific), pure eb (ratio=1.0), pure arith (ratio=0.0).
    pure_eb_test = os.path.join(data_root, _ratio_str(1.0), "test.csv")
    pure_arith_test = os.path.join(data_root, _ratio_str(0.0), "test.csv")
    for path, label in ((pure_eb_test, "entity-binding"), (pure_arith_test, "arithmetic")):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing pure {label} test CSV at {path}. Run 01_create_data.py "
                "with ratio 0.0 and 1.0 in config.data.mixing_ratios."
            )

    # Lazy model load — only when faithfulness is actually needed.
    model = None
    metric = None

    for method in methods:
        print(f"\n=== Evaluating method: {method} ===")
        method_eval_dir = os.path.join(eval_root, method)
        os.makedirs(method_eval_dir, exist_ok=True)

        results_json_path = os.path.join(method_eval_dir, "faithfulness_results.json")
        if os.path.exists(results_json_path) and not args.force:
            with open(results_json_path) as f:
                results = json.load(f)
            print(f"  loaded existing {results_json_path}")
        else:
            results = {}

        needs_save = False

        for ratio in all_ratios:
            rstr = _ratio_str(ratio)
            circuit_path = os.path.join(circuits_root, method, f"{rstr}.pt")
            if not os.path.exists(circuit_path):
                print(f"  skipping ratio={rstr}: missing {circuit_path}")
                continue

            mixed_test = os.path.join(data_root, rstr, "test.csv")
            if not os.path.exists(mixed_test):
                print(f"  skipping ratio={rstr}: missing mixed test {mixed_test}")
                continue

            existing = results.get(rstr)
            have_all = (
                existing is not None
                and "circuit_size" in existing
                and "faithfulness" in existing
                and all(t in existing["faithfulness"] for t in TEST_SETS)
            )
            if have_all and not args.force:
                print(f"  ratio={rstr}: already evaluated")
                continue

            # Circuit size is cheap and model-free — compute first.
            circuit_size = _circuit_size(circuit_path, GAMMA)

            # Faithfulness requires the model.
            if model is None:
                model, _ = load_model(config.model.name, config.model.cache_dir)
                model.tokenizer.pad_token = model.tokenizer.eos_token
                metric = metrics.get_logit_diff(model.tokenizer)

            faith_entry: Dict[str, dict] = {}
            test_paths = {
                "mixed": mixed_test,
                "entity-binding": pure_eb_test,
                "arithmetic": pure_arith_test,
            }
            for test_set, data_path in test_paths.items():
                print(
                    f"  [{method} | ratio={rstr} | test={test_set}] "
                    f"run_faithfulness on {data_path}"
                )
                r = run_faithfulness(model, circuit_path, data_path, metric)
                cpr, cmd = compute_cpr_cmd(r["faithfulness"])
                faith_entry[test_set] = {
                    "faithfulness_curve": r["faithfulness"],
                    "circuit_score_curve": r["circuit_score"],
                    "baseline_score": r["baseline_score"],
                    "corrupted_score": r["corrupted_score"],
                    "cpr": cpr,
                    "cmd": cmd,
                }

            results[rstr] = {
                "circuit_path": circuit_path,
                "circuit_size": circuit_size,
                "gamma": GAMMA,
                "faithfulness": faith_entry,
            }
            needs_save = True

        if needs_save:
            with open(results_json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  wrote {results_json_path}")

        if results:
            _write_summary_tables(method, method_eval_dir, results)


if __name__ == "__main__":
    main()
