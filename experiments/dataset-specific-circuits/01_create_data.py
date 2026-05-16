"""Generate mixed entity-binding + arithmetic train/val/test splits per mixing ratio.

For each ratio r in config.data.mixing_ratios, we build a dataset of size
n_data where r * n_data examples come from entity-binding and (1 - r) * n_data
come from arithmetic. Ratio = 1.0 is pure entity-binding; ratio = 0.0 is pure
arithmetic. Each row is tagged with its task_name and prompt_type.

Per-task pools are generated once (size n_data) to keep the examples used
across ratios consistent — ratio=0.3 draws the first 30% of the entity-binding
pool and the first 70% of the arithmetic pool, so sweeps are not noisy across
independent data draws.

Output layout:
    data/dataset-specific-circuits/{model_family}/{ratio}/{train,val,test}.csv
"""
import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from omegaconf import OmegaConf

from src.data.create_data import get_single_data
from src.utils.general_utils import load_model, model2family


def _build_task_pool(
    model,
    tokenizer,
    task_name: str,
    prompt_types: List[str],
    n_data: int,
    only_correct: bool,
) -> pd.DataFrame:
    """Generate a pool of n_data examples for a task, evenly across prompt_types."""
    n_per_type = n_data // len(prompt_types)
    remainder = n_data - n_per_type * len(prompt_types)

    dfs = []
    for i, prompt_type in enumerate(prompt_types):
        # Distribute remainder across the first few prompt_types.
        n_this = n_per_type + (1 if i < remainder else 0)
        print(
            f"  generating {n_this} examples: task={task_name}, "
            f"prompt_type={prompt_type}"
        )
        data = get_single_data(
            model,
            tokenizer,
            task_type=task_name,
            prompt_type=prompt_type,
            n_data=n_this,
            only_correct=only_correct,
        )
        df = pd.DataFrame(data)
        df["task_name"] = task_name
        df["prompt_type"] = prompt_type
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def _split_df(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Dict[str, pd.DataFrame]:
    n = len(df)
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)
    return {
        "train": df.iloc[:n_train].reset_index(drop=True),
        "val": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": df.iloc[n_train + n_val :].reset_index(drop=True),
    }


def _mix_split(
    eb_split: pd.DataFrame,
    arith_split: pd.DataFrame,
    ratio: float,
) -> pd.DataFrame:
    """Mix entity-binding and arithmetic splits at the given ratio."""
    target_n = len(eb_split)  # both splits are the same size (n_data per pool)
    n_eb = int(round(ratio * target_n))
    n_arith = target_n - n_eb
    n_eb = min(n_eb, len(eb_split))
    n_arith = min(n_arith, len(arith_split))
    parts = []
    if n_eb > 0:
        parts.append(eb_split.iloc[:n_eb])
    if n_arith > 0:
        parts.append(arith_split.iloc[:n_arith])
    if not parts:
        return eb_split.iloc[:0].copy()
    mixed = pd.concat(parts, ignore_index=True)
    return mixed.sample(frac=1, random_state=42).reset_index(drop=True)


def _ratio_str(ratio: float) -> str:
    return f"{ratio:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate CSVs even if they already exist",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model_family = model2family(config.model.name)
    output_root = os.path.join(config.paths.output_dir, model_family)
    os.makedirs(output_root, exist_ok=True)

    ratios = list(config.data.mixing_ratios)

    # Decide which ratio dirs are still missing before touching the model.
    pending_ratios = []
    for ratio in ratios:
        ratio_dir = os.path.join(output_root, _ratio_str(ratio))
        splits_exist = all(
            os.path.exists(os.path.join(ratio_dir, f"{split}.csv"))
            for split in ("train", "val", "test")
        )
        if splits_exist and not args.force:
            print(f"Skipping ratio={_ratio_str(ratio)}: splits already exist at {ratio_dir}")
        else:
            pending_ratios.append(ratio)

    if not pending_ratios:
        print("All ratios already exist. Use --force to regenerate.")
        shutil.copy(args.config, output_root)
        return

    tasks_cfg = OmegaConf.to_container(config.data.tasks, resolve=True)
    if set(tasks_cfg.keys()) != {"entity-binding", "arithmetic"}:
        raise ValueError(
            f"Expected tasks = {{entity-binding, arithmetic}}, got {set(tasks_cfg.keys())}"
        )

    model, tokenizer = load_model(config.model.name, config.model.cache_dir)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    n_data = int(config.data.n_data)
    only_correct = bool(config.data.only_correct)

    print(f"Generating entity-binding pool (n={n_data}) ...")
    eb_pool = _build_task_pool(
        model,
        tokenizer,
        "entity-binding",
        list(tasks_cfg["entity-binding"]["prompt_types"]),
        n_data,
        only_correct,
    )
    print(f"Generating arithmetic pool (n={n_data}) ...")
    arith_pool = _build_task_pool(
        model,
        tokenizer,
        "arithmetic",
        list(tasks_cfg["arithmetic"]["prompt_types"]),
        n_data,
        only_correct,
    )

    # Trim to the smaller of the two pools if a task failed to produce n_data.
    common_n = min(len(eb_pool), len(arith_pool))
    if common_n < n_data:
        print(
            f"WARNING: pools smaller than n_data ({len(eb_pool)} eb, "
            f"{len(arith_pool)} arith). Truncating to {common_n}."
        )
    eb_pool = eb_pool.iloc[:common_n].reset_index(drop=True)
    arith_pool = arith_pool.iloc[:common_n].reset_index(drop=True)

    eb_splits = _split_df(eb_pool, config.data.train_ratio, config.data.val_ratio)
    arith_splits = _split_df(arith_pool, config.data.train_ratio, config.data.val_ratio)

    for ratio in pending_ratios:
        ratio_dir = os.path.join(output_root, _ratio_str(ratio))
        os.makedirs(ratio_dir, exist_ok=True)
        for split in ("train", "val", "test"):
            mixed = _mix_split(eb_splits[split], arith_splits[split], ratio)
            csv_path = os.path.join(ratio_dir, f"{split}.csv")
            mixed.to_csv(csv_path, index=False)
        n_train = len(_mix_split(eb_splits["train"], arith_splits["train"], ratio))
        n_val = len(_mix_split(eb_splits["val"], arith_splits["val"], ratio))
        n_test = len(_mix_split(eb_splits["test"], arith_splits["test"], ratio))
        print(
            f"  → {ratio_dir}/ ratio={_ratio_str(ratio)} "
            f"(train={n_train}, val={n_val}, test={n_test})"
        )

    shutil.copy(args.config, output_root)
    print(f"Copied config to {output_root}/")


if __name__ == "__main__":
    main()
