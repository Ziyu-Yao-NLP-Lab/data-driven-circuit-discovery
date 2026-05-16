import pdb
import torch as t
from pathlib import Path
import os
import random
import numpy as np
import argparse

from src.utils import general_utils as gu
from src.data import sequence_completion, ioi, entity_binding, arithmetic, sequence_completion

def split_and_save_data(data, file_path: str, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """
    Split data into train/val/test sets and save to CSV files.
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        save_path = f"{file_path}-{split_name}.csv"
        gu.save_data_to_csv(split_data, save_path)
        print(f"Saved {len(split_data)} samples to {save_path}")
    
    return train_data, val_data, test_data


def get_single_data(model, tokenizer, task_type: str, prompt_type: str, n_data: int = 1000, only_correct: bool = True, random_seed: int = 32):
    if task_type == "ioi":
        if "mixed" in prompt_type:
            data_abba = ioi.create_data(model, tokenizer, "ABBA", n_data=n_data, only_correct=only_correct, random_seed=random_seed)
            data_baba = ioi.create_data(model, tokenizer, "BABA", n_data=n_data, only_correct=only_correct, random_seed=random_seed)
            data = data_abba[:n_data//2] + data_baba[:n_data//2]
        elif "3-person" in prompt_type:
            data_abba = ioi.create_data(model, tokenizer, "ABBA", n_data=n_data, only_correct=only_correct, random_seed=random_seed)
            data_baba = ioi.create_data(model, tokenizer, "BABA", n_data=n_data, only_correct=only_correct, random_seed=random_seed)
            io_first_data = ioi.create_n_person_data_from_original(model, tokenizer, data_abba, n_person=3, only_correct=only_correct, io_first=True)
            s_first_data = ioi.create_n_person_data_from_original(model, tokenizer, data_baba, n_person=3, only_correct=only_correct, io_first=False)
            data = io_first_data[:n_data//2] + s_first_data[:n_data//2]
        else:
            data = ioi.create_data(model, tokenizer, prompt_type, n_data=n_data, only_correct=only_correct, random_seed=random_seed)
            
    elif task_type == "entity-binding":
        data = entity_binding.create_data(model, tokenizer, prompt_type, n_data=n_data, only_correct=only_correct, random_seed=random_seed)

    elif task_type == "arithmetic":
        data = arithmetic.create_data(model, tokenizer, prompt_type, n_data=n_data, only_correct=only_correct, random_seed=random_seed)
    elif task_type == "sequence-completion":
        data = sequence_completion.create_data(model, tokenizer, prompt_type, n_data=n_data, only_correct=only_correct, random_seed=random_seed)
    else:
        raise ValueError(f"Task type {task_type} not supported.")
    return data


def get_data(model, tokenizer, task_prompts: dict, n_data: int = 1000, only_correct: bool = False, random_seed: int = 32, data_path_dir: str = None):
    model_name = gu.get_model_name(model)
    for task_type, prompt_types in task_prompts.items():
        for prompt_type in prompt_types:
            print(f"Creating {task_type} {prompt_type} data...")
            data = get_single_data(model, tokenizer, task_type, prompt_type, n_data, only_correct, random_seed)
            if only_correct:
                save_path = f"{data_path_dir}/{task_type}/{gu.model2family(model_name)}/correct/{prompt_type}"
            else:
                save_path = f"{data_path_dir}/{task_type}/{gu.model2family(model_name)}/both/{prompt_type}"
            split_and_save_data(data, save_path, train_ratio=0.6, val_ratio=0.2)
