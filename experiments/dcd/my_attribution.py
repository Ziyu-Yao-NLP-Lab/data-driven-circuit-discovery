import argparse
import os
import torch
import pickle
from functools import partial
import pdb
import pandas as pd

from transformer_lens import HookedTransformer, HookedTransformerConfig
from huggingface_hub import hf_hub_download

from eap.graph import Graph
from eap.attribute import attribute
from eap.attribute_node import attribute_node

from src.utils import general_utils as gu
from src.utils import metrics


def load_interpbench_model():
    hf_cfg = hf_hub_download("mib-bench/interpbench", filename="ll_model_cfg.pkl")
    hf_model = hf_hub_download("mib-bench/interpbench", subfolder="ioi_all_splits", filename="ll_model_100_100_80.pth")

    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        # Some cases in InterpBench have the config as a HookedTransformerConfig object instead of a dict
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict
    cfg.device = "cuda"

    # Enable evaluation mode in the IOI model; has a different config during training
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(hf_model, map_location="cuda"))
    return model

def run_localization_dataset(model, tokenizer, data_path: str, method: str = 'exact', ablation: str = 'patching', ig_steps: int = 5, batch_size: int = 5, save_dir: str = None, level: str = 'edge', max_examples: int = None):

    """Run the localization experiment.

    Args:
        model: Loaded TransformerLens model.
        tokenizer: Corresponding tokenizer.
        data_path: Path to the CSV file.
        method: Attribution method ('EAP-IG-inputs', 'EAP', or 'exact').
        ablation: Ablation type.
        ig_steps: Number of integrated gradient steps.
        batch_size: Batch size for the dataloader.
        save_dir: Directory to save the output .pt graph.
        level: Attribution level ('edge', 'node', or 'neuron').
        max_examples: If set, cap the number of examples used. Useful for
            'exact' (edge activation patching) which is very slow.
    """
    assert method in ['exact', 'EAP-IG-inputs', 'EAP'], "Method must be either exact, EAP-IG-inputs, or EAP."
    assert ablation in ['patching', 'zero', 'mean', 'mean-positional', 'optimal'], "Ablation must be either patching, zero, mean, mean-positional, or optimal."
    assert ig_steps > 0, "IG steps must be greater than 0."
    assert batch_size > 0, "Batch size must be greater than 0."

    neuron_level = level == "neuron"
    node_scores = level == "node"

    ds = gu.EAPDataset(data_path, n_samples=max_examples)
    dataloader = ds.to_dataloader(batch_size)
    attribution_metric = metrics.get_logit_diff(model.tokenizer)
    graph = Graph.from_model(model, neuron_level=neuron_level, node_scores=node_scores)
    if level == 'edge':
        attribute(model, graph, dataloader, attribution_metric, method, ablation, ig_steps=ig_steps, intervention_dataloader=dataloader)
    else:
        attribute_node(model, graph, dataloader, attribution_metric, method, ablation, neuron=neuron_level, ig_steps=ig_steps, intervention_dataloader=dataloader)
    # Save the graph
    method_name_saveable = f"-{method}"
    graph_path = f"{save_dir}/{data_path.split('/')[-1].replace('.csv', f'{method_name_saveable}.pt')}"
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    graph.to_pt(graph_path)
    print(f"Saved graph to {graph_path}")

def run_single_example_localization(model, tokenizer, data_path: str, method: str = 'EAP-IG-inputs', ablation: str = 'patching', ig_steps: int = 5, batch_size: int = 1, save_dir: str = None, level: str = 'edge', max_nan_skips: int = 10):

    """Run the localization experiment."""
    assert method in ['exact', 'EAP-IG-inputs'], "Method must be either exact or EAP-IG-inputs."
    assert ablation in ['patching', 'zero', 'mean', 'mean-positional', 'optimal'], "Ablation must be either patching, zero, mean, mean-positional, or optimal."
    assert ig_steps > 0, "IG steps must be greater than 0."
    assert batch_size > 0, "Batch size must be greater than 0."

    neuron_level = level == "neuron"
    node_scores = level == "node"
    my_data = pd.read_csv(data_path)
    skipped = []
    for i in range(len(my_data)):
        # Build output path early so we can skip already-completed examples
        method_name_saveable = f"-{i}-{method}"
        graph_path = f"{save_dir}/{data_path.split('/')[-1].replace('.csv', f'{method_name_saveable}.pt')}"
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        if os.path.exists(graph_path):
            print(f"Skipping train-{i} (already exists): {graph_path}")
            continue
        data = my_data.iloc[i].to_frame().T
        ds = gu.EAPDataset_from_data(data)
        dataloader = ds.to_dataloader(1)
        attribution_metric = metrics.get_logit_diff(model.tokenizer)
        graph = Graph.from_model(model, neuron_level=neuron_level, node_scores=node_scores)
        try:
            if level == 'edge':
                attribute(model, graph, dataloader, attribution_metric, method, ablation, ig_steps=ig_steps, intervention_dataloader=dataloader)
            else:
                attribute_node(model, graph, dataloader, attribution_metric, method, ablation, neuron=neuron_level, ig_steps=ig_steps, intervention_dataloader=dataloader)
        except ValueError as e:
            print(f"WARNING: Skipping train-{i} due to error: {e}")
            skipped.append(i)
            if len(skipped) > max_nan_skips:
                raise RuntimeError(
                    f"Too many NaN examples ({len(skipped)} > {max_nan_skips}). "
                    f"Skipped indices: {skipped}. "
                    "This likely indicates a systematic data problem — fix the data before rerunning."
                )
            continue
        graph.to_pt(graph_path)
        print(f"Saved graph to {graph_path}")
    if skipped:
        print(f"Attribution complete. Skipped {len(skipped)} examples due to NaN: {skipped}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--optimal-ablation-path", type=str, default=None)
    parser.add_argument("--level", type=str, choices=['node', 'neuron', 'edge'], default='edge')
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='train')
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--circuit-dir", type=str, default='circuits')
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs='+', required=True)
    parser.add_argument("--filter-types", type=str, nargs='+', choices=['correct', 'both'], default=['correct'])
    args = parser.parse_args()


    for model_name in args.models:
        model, _ = gu.load_model(model_name, cache_dir=args.cache_dir)
        attribution_metric = metrics.get_logit_diff(model.tokenizer)
        print(f"Loaded model {model_name}")
        neuron_level = args.level == "neuron"
        node_scores = args.level == "node"
        data_root_path = f"data/{args.tasks[0]}/{gu.model2family(model_name)}"

        for data_name in args.datasets:
            for filter_type in args.filter_types:
                print(f"Processing {data_name} with filter type {filter_type}")
                data_path = f"{data_root_path}/{filter_type}/{data_name}"
                my_data = pd.read_csv(data_path)
                # extract tenth row from my_data along with the header
                for i in range(len(my_data)):
                    data = my_data.iloc[i].to_frame().T
                    ds = gu.EAPDataset_from_data(data)
                    dataloader = ds.to_dataloader(1)
                    graph = Graph.from_model(model, neuron_level=neuron_level, node_scores=node_scores)
                    if args.level == 'edge':
                        attribute(model, graph, dataloader, attribution_metric, args.method, args.ablation, 
                                    ig_steps=args.ig_steps,
                                    intervention_dataloader=dataloader)
                    else:
                        attribute_node(model, graph, dataloader, attribution_metric, args.method, 
                                        args.ablation, neuron=args.level == 'neuron', ig_steps=args.ig_steps,
                                        intervention_dataloader=dataloader)
                    # Save the graph
                    method_name_saveable = f"{i}-{args.method}"
                    graph_path = data_path.replace("data/", "results/mechanism-aware-circuits/circuits/").replace(".csv", f"{method_name_saveable}.pt")
                    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
                    graph.to_pt(graph_path)
                    print(f"Saved graph to {graph_path}")
                
