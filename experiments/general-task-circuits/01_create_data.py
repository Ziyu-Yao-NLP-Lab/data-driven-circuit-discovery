"""Generate per-variant train/val/test data for the general-task-circuits experiment.

Each variant listed in config.data.variants is generated independently and
saved to data/general-task-circuits/{task_name}/{model_name}/{prompt_type}/
as train.csv, val.csv, test.csv. Unlike the DCD pipeline, prompt types are
NEVER mixed — the whole point of this experiment is to compare circuits
discovered on disjoint prompt-type variants.
"""
import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from omegaconf import OmegaConf

from src.data.create_data import get_single_data
from src.utils.general_utils import load_model


def split_df(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> List[pd.DataFrame]:
    """Split a DataFrame into train/val/test by the given ratios."""
    n = len(df)
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)
    return [
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val :].reset_index(drop=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate variants even if CSVs already exist",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    output_root = config.paths.output_dir
    os.makedirs(output_root, exist_ok=True)

    # Check which variants still need to be generated before loading the model.
    variants = list(config.data.variants)
    pending = []
    for variant in variants:
        prompt_type = variant.prompt_type
        variant_dir = os.path.join(output_root, prompt_type)
        splits_exist = all(
            os.path.exists(os.path.join(variant_dir, f"{split}.csv"))
            for split in ("train", "val", "test")
        )
        if splits_exist and not args.force:
            print(f"Skipping {prompt_type}: splits already exist at {variant_dir}")
        else:
            pending.append(variant)

    if not pending:
        print("All variants already exist. Use --force to regenerate.")
        shutil.copy(args.config, output_root)
        return

    model, tokenizer = load_model(config.model.name, config.model.cache_dir)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    for variant in pending:
        prompt_type = variant.prompt_type
        n_data = int(variant.n_data)
        print(f"Creating data for task={config.data.task_name}, prompt_type={prompt_type}, n_data={n_data}")

        data = get_single_data(
            model,
            tokenizer,
            task_type=config.data.task_name,
            prompt_type=prompt_type,
            n_data=n_data,
            only_correct=config.data.only_correct,
        )

        df = pd.DataFrame(data)
        df["prompt_type"] = prompt_type

        train_df, val_df, test_df = split_df(
            df, config.data.train_ratio, config.data.val_ratio
        )

        variant_dir = os.path.join(output_root, prompt_type)
        os.makedirs(variant_dir, exist_ok=True)
        train_df.to_csv(os.path.join(variant_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(variant_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(variant_dir, "test.csv"), index=False)
        print(
            f"  → {variant_dir}/ (train={len(train_df)}, val={len(val_df)}, test={len(test_df)})"
        )

    shutil.copy(args.config, output_root)
    print(f"Copied config to {output_root}/")


if __name__ == "__main__":
    main()
