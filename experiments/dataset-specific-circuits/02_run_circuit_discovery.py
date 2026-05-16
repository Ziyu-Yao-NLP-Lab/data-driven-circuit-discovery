"""Run edge-attribution circuit discovery on each mixed dataset for Experiment 2.

For every (mixing_ratio, method) pair in the config, run attribution on the
mixed train.csv and save the resulting graph to:
  results/dataset-specific-circuits/{model}/circuits/{method}/{ratio}.pt

The output is a *single* circuit per (ratio, method) — the claim of this
experiment is that hypothesis-driven methods will return one high-faithfulness
circuit for the mixed dataset regardless of mixing ratio.

Methods are named per the config:
- EAP-IG-inputs: integrated-gradient edge attribution (ig_steps > 1)
- EAP:           standard edge attribution (EAP-IG-inputs with ig_steps=1)
- EAct:          edge activation patching (maps to method='exact' under the hood)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from omegaconf import OmegaConf

from experiments.dcd.my_attribution import run_localization_dataset
from src.utils.general_utils import load_model, model2family
from src.utils import eap_patch

# NaN-tolerant get_scores_eap_ig patch so a single bad example does not kill
# a multi-hour attribution run. Must run before any `attribute` call.
eap_patch.apply()


METHOD_TO_IMPL = {
    "EAP-IG-inputs": "EAP-IG-inputs",
    "EAP": "EAP-IG-inputs",  # EAP = EAP-IG-inputs with ig_steps=1
    "EAct": "exact",
}


def parse_list(flag_value: Optional[str]) -> Optional[List[str]]:
    if flag_value is None:
        return None
    return [x.strip() for x in flag_value.split(",") if x.strip()]


def _ratio_str(ratio: float) -> str:
    return f"{ratio:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method names to run (subset of config methods)",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=None,
        help="Comma-separated mixing ratios to run (subset of config ratios)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run attribution even if the output .pt already exists",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model_family = model2family(config.model.name)

    method_filter = parse_list(args.methods)
    ratio_filter = parse_list(args.ratios)

    all_ratios = [float(r) for r in config.data.mixing_ratios]
    if ratio_filter is not None:
        keep = {float(r) for r in ratio_filter}
        all_ratios = [r for r in all_ratios if r in keep]

    all_methods = list(config.attribution.methods)
    if method_filter is not None:
        all_methods = [m for m in all_methods if m.name in method_filter]

    if not all_ratios or not all_methods:
        print("Nothing to run after filtering — check --methods / --ratios.")
        return

    results_root = f"results/dataset-specific-circuits/{model_family}"
    circuits_root = os.path.join(results_root, "circuits")
    os.makedirs(results_root, exist_ok=True)
    shutil.copy(args.config, results_root)

    data_root = os.path.join(config.paths.output_dir, model_family)

    # Plan the jobs before loading the model so we can early-exit if everything
    # is already on disk.
    jobs = []  # list of (ratio, method_cfg, data_path, target_path)
    for ratio in all_ratios:
        rstr = _ratio_str(ratio)
        data_path = os.path.join(data_root, rstr, "train.csv")
        if not os.path.exists(data_path):
            print(
                f"WARNING: missing {data_path} — skipping ratio={rstr}. "
                "Run 01_create_data.py first."
            )
            continue
        for method_cfg in all_methods:
            method_name = method_cfg.name
            method_dir = os.path.join(circuits_root, method_name)
            target_path = os.path.join(method_dir, f"{rstr}.pt")
            if os.path.exists(target_path) and not args.force:
                print(f"Skipping existing {target_path}")
                continue
            jobs.append((ratio, method_cfg, data_path, target_path))

    if not jobs:
        print("All circuits already exist. Use --force to re-run.")
        return

    model, tokenizer = load_model(config.model.name, config.model.cache_dir)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    ablation = config.attribution.ablation
    level = config.attribution.level
    batch_size = int(config.attribution.batch_size)

    for ratio, method_cfg, data_path, target_path in jobs:
        rstr = _ratio_str(ratio)
        method_name = method_cfg.name
        impl_method = METHOD_TO_IMPL[method_name]

        if "ig_steps" in method_cfg:
            ig_steps = int(method_cfg.ig_steps)
        elif method_name == "EAP":
            ig_steps = 1
        else:
            ig_steps = 5

        # 'exact' (EAct) ignores ig_steps but the helper asserts > 0.
        if impl_method == "exact":
            ig_steps = max(ig_steps, 1)

        max_examples = (
            method_cfg.get("max_examples", None) if hasattr(method_cfg, "get") else None
        )

        method_dir = os.path.dirname(target_path)
        os.makedirs(method_dir, exist_ok=True)

        print(
            f"[ratio={rstr} | {method_name}] attribution on {data_path} "
            f"(impl={impl_method}, ig_steps={ig_steps}, batch_size={batch_size})"
        )

        run_localization_dataset(
            model=model,
            tokenizer=tokenizer,
            data_path=data_path,
            method=impl_method,
            ablation=ablation,
            ig_steps=ig_steps,
            batch_size=batch_size,
            save_dir=method_dir,
            level=level,
            max_examples=max_examples,
        )

        # run_localization_dataset saves as "{data_basename}-{impl_method}.pt"
        # (e.g. train-EAP-IG-inputs.pt). Rename to "{ratio}.pt".
        saved_name = os.path.basename(data_path).replace(".csv", f"-{impl_method}.pt")
        saved_path = os.path.join(method_dir, saved_name)
        if not os.path.exists(saved_path):
            raise FileNotFoundError(
                f"Expected run_localization_dataset to save {saved_path}, "
                "but the file is missing."
            )
        os.replace(saved_path, target_path)
        print(f"  → {target_path}")


if __name__ == "__main__":
    main()
