"""Run edge-attribution circuit discovery on each variant for Experiment 1.

For every (variant, method) pair in the config, run attribution on the
variant's train.csv and save the resulting graph to:
  results/general-task-circuits/{task}/{model}/circuits/{method}/{prompt_type}.pt

Methods are named per the config:
- EAP-IG-inputs: integrated-gradient edge attribution (with ig_steps > 1)
- EAP:           standard edge attribution (implemented as EAP-IG-inputs with ig_steps=1)
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

# Install the NaN-tolerant get_scores_eap_ig patch so a single bad example
# does not kill a 45-minute attribution run. Must run before any `attribute`
# call (which happens inside `run_localization_dataset`).
eap_patch.apply()


# Map config method names to (underlying_method, ig_steps_default)
METHOD_TO_IMPL = {
    "EAP-IG-inputs": "EAP-IG-inputs",
    "EAP": "EAP-IG-inputs",  # EAP = EAP-IG-inputs with ig_steps=1
    "EAct": "exact",
}


def parse_list(flag_value: Optional[str]) -> Optional[List[str]]:
    if flag_value is None:
        return None
    return [x.strip() for x in flag_value.split(",") if x.strip()]


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
        "--variants",
        type=str,
        default=None,
        help="Comma-separated prompt_types to run (subset of config variants)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run attribution even if the output .pt already exists",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. attribution.batch_size=256",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    task = config.data.task_name
    model_family = model2family(config.model.name)

    method_filter = parse_list(args.methods)
    variant_filter = parse_list(args.variants)

    # Gather all (variant, method) pairs we need to run.
    all_variants = list(config.data.variants)
    if variant_filter is not None:
        all_variants = [v for v in all_variants if v.prompt_type in variant_filter]

    all_methods = list(config.attribution.methods)
    if method_filter is not None:
        all_methods = [m for m in all_methods if m.name in method_filter]

    if not all_variants or not all_methods:
        print("Nothing to run after filtering — check --methods / --variants.")
        return

    results_root = f"results/general-task-circuits/{task}/{model_family}"
    circuits_root = os.path.join(results_root, "circuits")
    os.makedirs(results_root, exist_ok=True)
    shutil.copy(args.config, results_root)

    # Plan: figure out which jobs still need running before loading the model.
    jobs = []  # list of (variant, method_cfg, data_path, target_path)
    for variant in all_variants:
        prompt_type = variant.prompt_type
        data_path = os.path.join(
            config.paths.output_dir, prompt_type, "train.csv"
        )
        if not os.path.exists(data_path):
            print(f"WARNING: missing {data_path} — skipping {prompt_type}. Run 01_create_data.py first.")
            continue
        for method_cfg in all_methods:
            method_name = method_cfg.name
            method_dir = os.path.join(circuits_root, method_name)
            target_path = os.path.join(method_dir, f"{prompt_type}.pt")
            if os.path.exists(target_path) and not args.force:
                print(f"Skipping existing {target_path}")
                continue
            jobs.append((variant, method_cfg, data_path, target_path))

    if not jobs:
        print("All circuits already exist. Use --force to re-run.")
        return

    model, tokenizer = load_model(config.model.name, config.model.cache_dir)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    ablation = config.attribution.ablation
    level = config.attribution.level
    batch_size = int(config.attribution.batch_size)

    for variant, method_cfg, data_path, target_path in jobs:
        prompt_type = variant.prompt_type
        method_name = method_cfg.name
        impl_method = METHOD_TO_IMPL[method_name]

        # ig_steps resolution: explicit per-method value; EAP defaults to 1.
        if "ig_steps" in method_cfg:
            ig_steps = int(method_cfg.ig_steps)
        elif method_name == "EAP":
            ig_steps = 1
        else:
            ig_steps = 5

        # For 'exact' (EAct) ig_steps is unused by the attribution code but
        # the helper asserts ig_steps > 0, so we keep a sane default.
        if impl_method == "exact":
            ig_steps = max(ig_steps, 1)

        max_examples = method_cfg.get("max_examples", None) if hasattr(method_cfg, "get") else None

        method_dir = os.path.dirname(target_path)
        os.makedirs(method_dir, exist_ok=True)

        print(
            f"[{prompt_type} | {method_name}] attribution on {data_path} "
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
        # (e.g. train-EAP-IG-inputs.pt). Rename to "{prompt_type}.pt".
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
