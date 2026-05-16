import argparse
import os
import shutil
import sys
from pathlib import Path

# Repo root (parent of experiments/) so `src` imports work when run as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from omegaconf import OmegaConf

from experiments.dcd.my_attribution import run_single_example_localization
from src.utils.general_utils import load_model


def validate_token_lengths(data_path: str, tokenizer) -> None:
    """Check that every example has matching clean/corrupted token lengths.

    Raises RuntimeError listing all mismatched row indices so the problem is
    visible before any GPU time is spent on attribution.
    """
    df = pd.read_csv(data_path)
    mismatches = []
    for i, row in df.iterrows():
        clean_len = len(tokenizer.encode(row["clean"]))
        corrupted_len = len(tokenizer.encode(row["corrupted"]))
        if clean_len != corrupted_len:
            mismatches.append((i, clean_len, corrupted_len))

    if mismatches:
        details = "\n".join(
            f"  row {i}: clean={c}, corrupted={p}" for i, c, p in mismatches
        )
        raise RuntimeError(
            f"Token length mismatches found in {data_path} ({len(mismatches)} rows):\n{details}\n"
            "Fix the data generation before running attribution."
        )
    print(f"Validation passed: all {len(df)} examples have matching token lengths.")


def main() -> None:
    """Run per-example edge attribution on the DCD training split.

    Reads the YAML config (same file used for data creation), loads the model,
    and calls run_single_example_localization() on the training CSV.  Per-example
    .pt graph files are written to
    results/dcd/{task_name}/{model_name}/circuits/raw_edge_scores/.
    The config and training CSV are copied into the results directory for
    reproducibility.

    Accepts OmegaConf dotlist overrides after --config, e.g.:
        attribution.max_examples=5
    """
    parser = argparse.ArgumentParser(
        description="Run EAP attribution on DCD training data.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. attribution.max_examples=5",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    # Derive {model_name} directory token from the data output path so it stays
    # consistent with how 01_create_data.py named things (e.g. qwen2.5-7b-instruct).
    task_name: str = config.data.task_name
    model_dir: str = Path(config.paths.output_dir).name  # last component

    results_root = Path(f"results/dcd/{task_name}/{model_dir}")
    save_dir = results_root / "circuits" / "raw_edge_scores"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy config into results directory for reproducibility
    shutil.copy(args.config, results_root)
    print(f"Copied config to {results_root}/")

    model, tokenizer = load_model(config.model.name, config.model.cache_dir)

    attr_cfg = config.attribution
    train_csv = os.path.join(config.paths.output_dir, "train.csv")

    # If max_examples is set, write a sliced CSV into the results dir so that
    # run_single_example_localization receives a real file path and output filenames
    # stay consistent (train-{i}-{method}.pt).
    max_examples = OmegaConf.select(config, "attribution.max_examples")
    if max_examples is not None:
        sliced_csv = str(results_root / "train_head.csv")
        pd.read_csv(train_csv).head(int(max_examples)).to_csv(sliced_csv, index=False)
        data_path = sliced_csv
        print(f"Capped to {max_examples} examples; sliced CSV at {sliced_csv}")
    else:
        data_path = train_csv

    print(f"Validating token lengths in {data_path}...")
    validate_token_lengths(data_path, tokenizer)

    print(f"Running attribution on {data_path}")

    run_single_example_localization(
        model=model,
        tokenizer=tokenizer,
        data_path=data_path,
        method=attr_cfg.method,
        ablation=attr_cfg.ablation,
        ig_steps=attr_cfg.ig_steps,
        batch_size=attr_cfg.batch_size,
        save_dir=str(save_dir),
        level=attr_cfg.level,
    )

    data_dir = results_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(train_csv, data_dir / "train.csv")
    print(f"Copied training data to {data_dir}/")
    print(f"Attribution complete. Graphs saved to {save_dir}/")


if __name__ == "__main__":
    main()
