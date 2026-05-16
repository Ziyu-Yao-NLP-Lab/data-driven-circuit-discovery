import argparse
import math
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

from src.data.create_data import get_single_data
from src.utils.general_utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    model, tokenizer = load_model(config.model.name, config.model.cache_dir)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    train_dfs, val_dfs, test_dfs = [], [], []

    task_prompt_pairs = [
        (task_name, prompt_type)
        for task_name, task_cfg in config.data.tasks.items()
        for prompt_type in task_cfg.prompt_types
    ]
    n_per_type = config.data.n_data // len(task_prompt_pairs)

    for task_name, prompt_type in task_prompt_pairs:
        print(f"Creating data for task: {task_name}, prompt_type: {prompt_type}")
        data = get_single_data(
            model,
            tokenizer,
            task_type=task_name,
            prompt_type=prompt_type,
            n_data=n_per_type,
            only_correct=config.data.only_correct,
        )

        df = pd.DataFrame(data)
        df["prompt_type"] = prompt_type

        n = len(df)
        n_train = math.floor(n * config.data.train_ratio)
        n_val = math.floor(n * config.data.val_ratio)

        train_dfs.append(df.iloc[:n_train])
        val_dfs.append(df.iloc[n_train : n_train + n_val])
        test_dfs.append(df.iloc[n_train + n_val :])

    out_dir = config.paths.output_dir
    os.makedirs(out_dir, exist_ok=True)

    pd.concat(train_dfs).reset_index(drop=True).to_csv(
        os.path.join(out_dir, "train.csv"), index=False
    )
    pd.concat(val_dfs).reset_index(drop=True).to_csv(
        os.path.join(out_dir, "val.csv"), index=False
    )
    pd.concat(test_dfs).reset_index(drop=True).to_csv(
        os.path.join(out_dir, "test.csv"), index=False
    )
    print(f"Saved train/val/test splits to {out_dir}/")

    shutil.copy(args.config, out_dir)
    print(f"Copied config to {out_dir}/")


if __name__ == "__main__":
    main()
