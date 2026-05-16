"""
This file contains utility functions for the circuit composition experiments,
including data generation, model loading, representation extraction, and other
miscellaneous helpers.
"""

import os
import pickle
import random
import smtplib
from email.mime.text import MIMEText
from typing import List, Tuple
import pandas as pd
import pdb

import torch
import transformer_lens as lens
from tqdm import tqdm


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_name: str, cache_dir: str, device: str = "cuda"):
    """Loads a HookedTransformer model from a short name."""
    print(f"Loading model: {model_name}...")
    if model_name == "llama3-8b":
        model_name_hf = "meta-llama/Meta-Llama-3-8B"
    elif model_name == "llama3.1-8b":
        model_name_hf = "meta-llama/Llama-3.1-8B"
    elif model_name == "Llama-3.1-8B-Instruct":
        model_name_hf = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_name == "pythia":
        model_name_hf = "EleutherAI/pythia-6.9b"
    elif model_name == "gpt-j":
        model_name_hf = "EleutherAI/gpt-j-6b"
    else:
        model_name_hf = model_name  # Assume it's a full HF path

    model = lens.HookedTransformer.from_pretrained(
        model_name_hf,
        cache_dir=cache_dir,
        fold_ln=True,
        center_unembed=True,
        center_writing_weights=True,
        device=device,
    )
    print("Model loaded.")
    return model

# ============================================================================
# Localization utilities
# ============================================================================

def save_data_to_csv(data, save_path: str):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    df = pd.DataFrame.from_dict(data)
    df = df.sample(frac=1)
    df.to_csv(f'{save_path}', index=False)

def get_single_token_numbers(model):
    single_token_numbers = []
    single_token_ids = []
    
    for num in range(1000):
        token_ids = model.tokenizer.encode(str(num), add_special_tokens=False)
        if len(token_ids) == 1:
            single_token_numbers.append(num)
            single_token_ids.append(token_ids[0])
    
    return single_token_numbers, single_token_ids

def prepare_verbal_localization_data(raw_data, model):
    single_tokens, single_tokens_ids = get_single_token_numbers(model)
    data = []
    for idx in range(len(raw_data)):
        tmp = {}
        tmp['clean'] = raw_data[idx][0]
        if idx == len(raw_data)-1:
            tmp['corrupted'] = raw_data[idx-1][0]
        else:
            tmp['corrupted'] = raw_data[idx+1][0]
        tmp['label'] = raw_data[idx][1]
        tmp['correct_idx'] = model.tokenizer.encode(str(tmp['label']), add_special_tokens=False)[0]
        tmp['incorrect_idx'] = [token_id for token_id in single_tokens_ids 
                            if token_id != tmp['correct_idx']]
        
        assert tmp['correct_idx'] not in tmp['incorrect_idx'], f"correct_idx {tmp['correct_idx']} found in incorrect idx list"

        tmp['corrupted_labels'] = [token for token in single_tokens 
                               if str(token) != str(tmp['label'])]
        data.append(tmp)
    return data

def prepare_localization_data(raw_data, model):
    single_tokens, single_tokens_ids = get_single_token_numbers(model)
    # Pre-compute token lengths to avoid redundant tokenization in the search loop
    token_lengths = [len(model.tokenizer.tokenize(item[0])) for item in raw_data]
    data = []
    for idx in range(len(raw_data)):
        tmp = {}
        tmp['clean'] = raw_data[idx][0]
        clean_len = token_lengths[idx]
        if idx == len(raw_data)-1:
            selected_idx = idx - 1
            while selected_idx >= 0 and token_lengths[selected_idx] != clean_len:
                selected_idx -= 1
            if selected_idx < 0:
                continue
            tmp['corrupted'] = raw_data[selected_idx][0]
        else:
            selected_idx = idx + 1
            while selected_idx < len(raw_data) and token_lengths[selected_idx] != clean_len:
                selected_idx += 1
            if selected_idx >= len(raw_data):
                continue
            tmp['corrupted'] = raw_data[selected_idx][0]
        
        if len(model.tokenizer.tokenize(tmp['corrupted'])) != len(model.tokenizer.tokenize(tmp['clean'])):
            print(f"corrupted {tmp['corrupted']} and clean {tmp['clean']} have different token lengths at index {idx}")
            continue
        
        tmp['correct_idx'] = model.tokenizer.encode(str(raw_data[idx][1]), add_special_tokens=False)[0]
        try:
            assert tmp['correct_idx'] in single_tokens_ids, f"correct_idx {tmp['correct_idx']} at index {idx} not found in single_tokens_ids"
        except:
            print(f"correct_idx {tmp['correct_idx']} at index {idx} not found in single_tokens_ids")
        tmp['incorrect_idx'] = [token_id for token_id in single_tokens_ids 
                            if token_id != tmp['correct_idx']]
        assert tmp['correct_idx'] not in tmp['incorrect_idx'], f"correct_idx {tmp['correct_idx']} found in incorrect idx list"

        tmp['label'] = raw_data[idx][1]
        tmp['corrupted_labels'] = [token for token in single_tokens 
                               if str(token) != str(tmp['label'])]
        data.append(tmp)

    return data

# ============================================================================
# Data Generation
# ============================================================================

def make_two_operand_prompts(
    model,
    num_prompts: int,
    max_op: int,
    max_answer_value: int,
    min_answer_value: int = 0,
    operator: str = "+",
    correct_only: bool = True
):
    """
    Generates two-operand prompts (e.g., "A+B=") and their corresponding
    answers, optionally filtering for those the model answers correctly.
    """
    prompts_and_answers = []
    expressions = []
    for op1 in tqdm(range(1, max_op), desc="Generating prompts", leave=False):
        for op2 in range(1, max_op):
            result = None
            prompt = None
            if operator == '+':
                result = op1 + op2
                prompt = f"{op1}+{op2}="
            elif operator == '-':
                result = op1 - op2
                prompt = f"{op1}-{op2}="
            elif operator == '*':
                result = op1 * op2
                prompt = f"{op1}*{op2}="
            elif operator == '/':
                if op2 == 0: continue
                if op1 % op2 == 0:
                    result = op1 // op2
                    prompt = f"{op1}/{op2}="
                else:
                    continue
            else:
                raise ValueError(f"Unsupported operator: {operator}")
            
            if prompt and 0 <= result < max_answer_value and result >= min_answer_value:
                expressions.append(prompt)

    if not expressions:
        raise ValueError("No valid expressions generated.")

    random.shuffle(expressions)

    pbar = tqdm(total=num_prompts, desc="Finding correct prompts")
    if not correct_only:
        # Just take the first `num_prompts` and get their answers
        for prompt in expressions[:num_prompts]:
            correct_answer = str(eval(prompt.rstrip("=")))
            prompts_and_answers.append((prompt, correct_answer))
            pbar.update(1)
        pbar.close()
        return prompts_and_answers

    # Filter for correct answers
    correct_count = 0
    index = 0
    while correct_count < num_prompts:
        if index >= len(expressions):
            # raise ValueError("Could not find enough correct prompts.")
            print(f"Using all prompts ({len(expressions)})")
            return prompts_and_answers
        
        prompt = expressions[index]
        index += 1
        if model.to_tokens(prompt, prepend_bos=True).shape[1] != 5:
            continue
        
        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model(tokens, return_type='logits')
        pred_id = logits[:, -1, :].argmax(dim=-1)
        predicted_token = model.to_str_tokens(pred_id.unsqueeze(0))[0].strip()

        try:
            correct_answer = eval(prompt.rstrip("="))
            if int(predicted_token) == correct_answer:
                correct_count += 1
                prompts_and_answers.append((prompt, str(correct_answer)))
                pbar.update(1)
        except (ValueError, SyntaxError):
            continue

    pbar.close()
    return prompts_and_answers


def make_pa_second_2_operand(
    model, num_prompts: int, max_op: int, max_answer_value: int, real_token: str, operator: str
):
    """
    Generates prompts for representation extraction on the second operand of a
    two-operand expression (e.g., "A+[B]"). No filtering is applied.
    """
    prompts_and_answers = []
    expressions = []
    try:
        real_value = int(real_token)
    except ValueError:
        raise ValueError("real_token must be convertible to an integer.")

    for op1 in range(1, max_op):
        result, prompt = None, None
        if operator == '+':
            result, prompt = op1 + real_value, f"{op1}+{real_token}"
        elif operator == '-':
            result, prompt = op1 - real_value, f"{op1}-{real_token}"
        elif operator == '*':
            result, prompt = op1 * real_value, f"{op1}*{real_token}"
        elif operator == '/':
            if real_value == 0: continue
            if op1 % real_value == 0:
                result, prompt = op1 // real_value, f"{op1}/{real_token}"
            else:
                continue
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        if prompt and 0 <= result < max_answer_value + max_op:
            expressions.append(prompt)

    if not expressions:
        raise ValueError("No valid expressions generated.")

    random.shuffle(expressions)
    
    for prompt in expressions[:num_prompts]:
        prompts_and_answers.append((prompt, "dummy_answer")) # Answers not needed

    return prompts_and_answers

def make_three_operand_prompts(
    model,
    num_prompts: int,
    max_op: int,
    max_answer_value: int,
    template: str,
    correct_only: bool = True
):
    """
    Generates three-operand prompts (e.g., "A+B-C=") and their corresponding
    answers, optionally filtering for those the model answers correctly.
    """
    prompts_and_answers = []
    expressions = []
    
    # Extract operators from template
    parts = template.replace("=","").strip().split(" ")
    op1_char = parts[1]
    op2_char = parts[3]

    if not correct_only:
        # Avoid building the full O(max_op^3) list — sample randomly instead
        ops = list(range(1, max_op))
        pbar = tqdm(total=num_prompts, desc="Generating prompts")
        attempts = 0
        max_attempts = num_prompts * 100
        seen = set()
        while len(prompts_and_answers) < num_prompts and attempts < max_attempts:
            opA, opB, opC = random.choice(ops), random.choice(ops), random.choice(ops)
            key = (opA, opB, opC)
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
            result = eval(f"{opA} {op1_char} {opB} {op2_char} {opC}")
            if 0 <= result < max_answer_value:
                prompt = template.format(A=opA, B=opB, C=opC)
                correct_answer = str(result)
                prompts_and_answers.append((prompt, correct_answer))
                pbar.update(1)
            attempts += 1
        pbar.close()
        if not prompts_and_answers:
            raise ValueError("No valid expressions generated.")
        return prompts_and_answers

    for opA in tqdm(range(1, max_op), desc="Generating prompts", leave=False):
        for opB in range(1, max_op):
            for opC in range(1, max_op):
                expr_str = f"{opA} {op1_char} {opB} {op2_char} {opC}"
                result = eval(expr_str)

                if 0 <= result < max_answer_value:
                    prompt = template.format(A=opA, B=opB, C=opC)
                    expressions.append(prompt)

    if not expressions:
        raise ValueError("No valid expressions generated.")

    random.shuffle(expressions)

    pbar = tqdm(total=num_prompts, desc="Finding correct prompts")

    # Filter for correct answers
    correct_count = 0
    index = 0
    while correct_count < num_prompts:
        if index >= len(expressions):
            print(f"Using all prompts ({len(expressions)})")
            return prompts_and_answers
            # raise ValueError(f"Could not find enough correct prompts (found {correct_count}).")
        
        prompt = expressions[index]
        index += 1
        
        # Ensure consistent tokenization length
        if model.to_tokens(prompt, prepend_bos=True).shape[1] != 10:
            continue
        
        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model(tokens, return_type='logits')
        pred_id = logits[:, -1, :].argmax(dim=-1)
        predicted_token = model.to_str_tokens(pred_id.unsqueeze(0))[0].strip()

        try:
            correct_answer = eval(prompt.split("=")[0].strip())
            if int(predicted_token) == correct_answer:
                correct_count += 1
                prompts_and_answers.append((prompt, str(correct_answer)))
                pbar.update(1)
        except (ValueError, SyntaxError):
            continue

    pbar.close()
    return prompts_and_answers


def make_pa_second_3_operand(
    model, num_prompts: int, max_op: int, max_answer_value: int, real_token: str, op1_char: str, op2_char: str
):
    """
    Generates prompts for representation extraction on the second operand of a
    three-operand expression (e.g., "A+[B]-C="). No filtering.
    This is for a fixed template of `op1 {op1_char} {real_token} {op2_char} 1 =`
    """
    prompts_and_answers = []
    expressions = []
    try:
        real_val_int = int(real_token)
    except ValueError:
        raise ValueError("real_token must be convertible to an int")

    for op1 in range(1, max_op):
        # The template is now flexible to the task's operators
        expr_str = f"{op1} {op1_char} {real_val_int} "
        result = eval(expr_str)
        prompt = f"{expr_str}"
        
        # if 0 <= result < max_answer_value + max_op:
        expressions.append(prompt)

    if not expressions:
        raise ValueError(f"No valid expressions found for make_pa_second_3_operand, max_op: {max_op}, max_answer_value: {max_answer_value}, real_token: {real_token}, op1_char: {op1_char}, op2_char: {op2_char}")

    random.shuffle(expressions)
    for prompt in expressions[:num_prompts]:
        prompts_and_answers.append((prompt, "dummy_answer"))

    return prompts_and_answers


def make_pa_third_3_operand(
    model, num_prompts: int, max_op: int, max_answer_value: int, real_token: str, op1_char: str, op2_char: str
):
    """
    Generates prompts for representation extraction on the third operand of a
    three-operand expression (e.g., "A+B-[C]="). No filtering.
    """
    prompts_and_answers, expressions = [], []
    try:
        real_val_int = int(real_token)
    except ValueError:
        raise ValueError("real_token must be convertible to int")

    for op1 in range(1, max_op):
        for op2 in range(1, max_op):
            expr_str = f"{op1} {op1_char} {op2} {op2_char} {real_val_int}"
            result = eval(expr_str)
            prompt = f"{op1} {op1_char} {op2} {op2_char} {real_token} = "
            expressions.append(prompt)

    if not expressions:
        raise ValueError("No valid expressions generated in make_pa_third_3_operand")
    
    random.shuffle(expressions)
    
    # Collect prompts without correctness filtering
    idx = 0
    while len(prompts_and_answers) < min(num_prompts, len(expressions)):
        if idx >= len(expressions):
            break
        prompt = expressions[idx]
        idx += 1
        if model.to_tokens(prompt, prepend_bos=True).shape[1] != 10:
            continue
        prompts_and_answers.append((prompt, "dummy_answer"))
        
    return prompts_and_answers

# ============================================================================
# Randomized CAMA Prompt Generation
# ============================================================================

def make_pa_second_2_operand_randomized(
    model, num_prompts: int, max_op: int, max_answer_value: int, real_token: str
) -> List[Tuple[str, str]]:
    """
    Generates prompts for representation extraction on the second operand of a
    two-operand expression, sampling randomly from all four operators.
    """
    prompts_and_answers = []
    expressions = []
    real_value = int(real_token)

    for op1 in range(1, max_op):
        for op_char in ["+", "-", "*", "/"]:
            prompt, result = None, None
            if op_char == '+':
                result, prompt = op1 + real_value, f"{op1}+{real_token}"
            elif op_char == '-':
                result, prompt = op1 - real_value, f"{op1}-{real_token}"
            elif op_char == '*':
                result, prompt = op1 * real_value, f"{op1}*{real_token}"
            elif op_char == '/':
                if real_value == 0: continue
                if op1 % real_value == 0:
                    result, prompt = op1 // real_value, f"{op1}/{real_token}"
                else: continue
            
            if prompt and 0 <= result < max_answer_value + max_op:
                expressions.append(prompt)

    if not expressions:
        raise ValueError("No valid expressions generated.")

    random.shuffle(expressions)
    
    for prompt in expressions[:num_prompts]:
        # The model's predicted token is not needed here, just the prompt
        prompts_and_answers.append((prompt, "dummy_answer"))

    return prompts_and_answers


def make_pa_third_2_operand_randomized(
    model, num_prompts: int, max_op: int, max_answer_value: int
) -> List[Tuple[str, str]]:
    """
    Generates prompts for representation extraction on the third token group (=)
    of a two-operand expression, sampling from all four operators.
    """
    prompts_and_answers = []
    expressions = []

    for op1 in range(1, max_op):
        for op2 in range(1, max_op):
            for op_char in ["+", "-", "*", "/"]:
                prompt, result = None, None
                if op_char == '+':
                    result, prompt = op1 + op2, f"{op1}+{op2}="
                elif op_char == '-':
                    result, prompt = op1 - op2, f"{op1}-{op2}="
                elif op_char == '*':
                    result, prompt = op1 * op2, f"{op1}*{op2}="
                elif op_char == '/':
                    if op2 == 0: continue
                    if op1 % op2 == 0:
                        result, prompt = op1 // op2, f"{op1}/{op2}="
                    else: continue

                if prompt and 0 <= result < max_answer_value:
                    expressions.append(prompt)
    
    if not expressions:
        raise ValueError("No valid expressions generated.")

    random.shuffle(expressions)
    for prompt in expressions[:num_prompts]:
        prompts_and_answers.append((prompt, "dummy_answer"))
    
    return prompts_and_answers


def make_pa_third_3_operand_randomized(
    model, num_prompts: int, max_op: int, max_answer_value: int, real_token: str
) -> List[Tuple[str, str]]:
    """
    Generates prompts for representation extraction on the third operand of a
    three-operand expression, sampling from all four ± combinations.
    """
    prompts_and_answers, expressions = [], []
    real_val_int = int(real_token)

    for op1 in range(1, max_op):
        for op2 in range(1, max_op):
            for op1_char in ["+", "-"]:
                for op2_char in ["+", "-"]:
                    expr_str = f"{op1} {op1_char} {op2} {op2_char} {real_val_int}"
                    result = eval(expr_str)
                    if 0 <= result < max_answer_value:
                        prompt = f"{op1} {op1_char} {op2} {op2_char} {real_token} = "
                        expressions.append(prompt)
    
    random.shuffle(expressions)
    for prompt in expressions[:num_prompts]:
        if model.to_tokens(prompt, prepend_bos=True).shape[1] == 10:
            prompts_and_answers.append((prompt, "dummy"))
    
    return prompts_and_answers

def make_pa_fourth_3_operand_randomized(
    model, num_prompts: int, max_op: int, max_answer_value: int
) -> List[Tuple[str, str]]:
    """
    Generates prompts for representation extraction on the fourth token group (=)
    of a three-operand expression, sampling from all four ± combinations.
    """
    prompts_and_answers, expressions = [], []

    for op1 in range(1, max_op):
        for op2 in range(1, max_op):
            for op3 in range(1, max_op):
                for op1_char in ["+", "-"]:
                    for op2_char in ["+", "-"]:
                        expr_str = f"{op1} {op1_char} {op2} {op2_char} {op3}"
                        result = eval(expr_str)
                        if 0 <= result < max_answer_value:
                            prompt = f"{op1} {op1_char} {op2} {op2_char} {op3} = "
                            expressions.append(prompt)
    
    random.shuffle(expressions)
    for prompt in expressions[:num_prompts]:
        if model.to_tokens(prompt, prepend_bos=True).shape[1] == 10:
            prompts_and_answers.append((prompt, "dummy"))

    return prompts_and_answers

# ============================================================================
# Custom Template Generation
# ============================================================================

def make_custom_prompts(
    model,
    num_prompts: int,
    max_op: int,
    max_answer_value: int,
    template: str,
    operator: str,
    correct_only: bool = True,
):
    """
    Generates prompts from a custom template string (e.g., for QA, word problems),
    optionally filtering for those the model answers correctly.
    """
    prompts_and_answers = []
    expressions = []
    for op1 in tqdm(range(1, max_op), desc="Generating prompts", leave=False):
        for op2 in range(1, op1): # Avoid op2 > op1 for subtraction
            result = eval(f"{op1} {operator} {op2}")
            if 0 <= result < max_answer_value:
                prompt = template.format(op1=op1, op2=op2)
                expressions.append((prompt, result))

    if not expressions:
        raise ValueError("No valid expressions were generated.")

    print(F"shuffling {len(expressions)} expressions...")
    random.shuffle(expressions)
    
    if not correct_only:
        for prompt, result in expressions[:num_prompts]:
            prompts_and_answers.append((prompt, str(result)))
        return prompts_and_answers

    print("Filtering for correct answers...")
    pbar = tqdm(total=num_prompts, desc="Finding correct prompts")
    correct_count = 0
    index = 0
    while correct_count < num_prompts:
        if index >= len(expressions):
            raise ValueError(f"Could not find enough correct prompts (found {correct_count}).")
        
        prompt, result = expressions[index]
        index += 1
        
        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model(tokens, return_type='logits')
        pred_id = logits[:, -1, :].argmax(dim=-1)
        predicted_token = model.to_str_tokens(pred_id.unsqueeze(0))[0].strip()

        try:
            if int(predicted_token) == result:
                correct_count += 1
                prompts_and_answers.append((prompt, str(result)))
                pbar.update(1)
        except (ValueError, SyntaxError):
            continue
            
    pbar.close()
    return prompts_and_answers


def make_pa_second_custom(
    num_prompts: int,
    max_op: int,
    max_answer_value: int,
    real_token: str,
    template: str,
    operator: str,
):
    """
    Generates prompts for representation extraction on the second operand for
    a custom template. No filtering is applied.
    """
    prompts_and_answers = []
    expressions = []
    try:
        real_value = int(real_token)
    except ValueError:
        raise ValueError("real_token must be convertible to an integer.")

    for op1 in range(1, max_op):
        result = eval(f"{op1} {operator} {real_value}")
        if 0 <= result < max_answer_value:
            prompt = template.format(op1=op1, op2=real_value)
            expressions.append(prompt)

    if not expressions:
        raise ValueError("No valid expressions were generated.")
        
    random.shuffle(expressions)
    
    for prompt in expressions[:num_prompts]:
        prompts_and_answers.append((prompt, real_token))

    return prompts_and_answers

# ============================================================================
# Representation Extraction
# ============================================================================

def get_intermediate_embedding_representations(model, dataset, token_positions):
    """
    Collects layer-wise representations for specified token positions.
    Returns a list of length n_layers, where each element is a tensor of shape
    (len(token_positions), hidden_dim).
    """
    reps_per_layer = []
    n_layers = model.cfg.n_layers
    
    # Pass only prompts to the inner function
    prompts = [item[0] for item in dataset]

    for layer in tqdm(range(n_layers), desc=f"Collecting reps for pos {token_positions}", leave=False):
        layer_rep = get_intermediate_token_embeddings(
            model, prompts, token_positions, layer
        )
        reps_per_layer.append(layer_rep)

    torch.cuda.empty_cache()
    return reps_per_layer


def get_intermediate_token_embeddings(model, dataset, token_positions, layer_idx):
    """
    Collects and averages representations for multiple token positions over a dataset.
    Returns a tensor of shape (len(token_positions), hidden_dim) for one layer.
    """
    accumulated_reps = [[] for _ in range(len(token_positions))]

    def collect_layer_representation(tensor, hook):
        for i, pos in enumerate(token_positions):
            rep_for_position = tensor[:, pos, :].mean(dim=0)
            accumulated_reps[i].append(rep_for_position.detach().clone())
        return tensor

    hooks = [(f"blocks.{layer_idx}.hook_resid_post", collect_layer_representation)]
    batch_size = 20
    
    with model.hooks(hooks), torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Layer {layer_idx}", leave=False):
            batch = dataset[i: i + batch_size]
            batch_tokens = torch.stack([
                model.to_tokens(prompt, prepend_bos=True)[0] for prompt in batch
            ])
            model(batch_tokens)

    final_means = [torch.stack(reps, dim=0).mean(dim=0) for reps in accumulated_reps]
    return torch.stack(final_means, dim=0)


# ============================================================================
# Data Handling
# ============================================================================

class ListDataset(torch.utils.data.Dataset):
    """A simple dataset class that wraps a list of data."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================================
# Miscellaneous Utilities
# ============================================================================

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def create_data(model, tokenizer, dataset_type, n_data, only_correct=True, random_seed: int = 32):
    # Pin global random state so random.shuffle / random.choice inside the
    # make_* helpers are deterministic across runs.
    random.seed(random_seed)
    if dataset_type == "2-operand-addition":
        max_op = 500
        max_answer_value = 999
        operator = "+"
        data = make_two_operand_prompts(
            model=model,
            num_prompts=n_data,
            max_op=max_op,
            max_answer_value=max_answer_value,
            operator=operator,
            correct_only=only_correct
        )
        
    elif dataset_type == "3-operand-addition":
        max_op = 500
        max_answer_value = 999
        operator = "+"
        template = "{A} + {B} + {C} = "
        data = make_three_operand_prompts(
            model=model,
            num_prompts=n_data,
            max_op=max_op,
            max_answer_value=max_answer_value,
            template=template,
            correct_only=only_correct
        )
    elif dataset_type == "100-300-2-operand-addition":
        max_op = 500
        max_answer_value = 300  
        min_answer_value = 100
        operator = "+"
        data = make_two_operand_prompts(
            model=model,
            num_prompts=n_data,
            max_op=max_op,
            max_answer_value=max_answer_value,
            min_answer_value=min_answer_value,
            operator=operator,
            correct_only=only_correct
        )
    elif dataset_type == "300-600-2-operand-addition":
        max_op = 500
        max_answer_value = 600
        min_answer_value = 300
        operator = "+"
        data = make_two_operand_prompts(
            model=model,
            num_prompts=n_data,
            max_op=max_op,
            max_answer_value=max_answer_value,
            min_answer_value=min_answer_value,
            operator=operator,
            correct_only=only_correct
        )
    elif dataset_type == "600-900-2-operand-addition":
        max_op = 500
        max_answer_value = 900
        min_answer_value = 600
        operator = "+"
        data = make_two_operand_prompts(
            model=model,
            num_prompts=n_data,
            max_op=max_op,
            max_answer_value=max_answer_value,
            min_answer_value=min_answer_value,
            operator=operator,
            correct_only=only_correct
        )
    elif dataset_type == "2-operand-addition-verbal-v1":
        max_op = 500
        max_answer_value = 999
        operator = "+"
        template = "Question: What is the addition of {op1} and {op2}? Answer: "
        data = make_custom_prompts(
            model=model,
            num_prompts=n_data,
            max_op=max_op,
            max_answer_value=max_answer_value,
            operator=operator,
            template=template,
            correct_only=only_correct
        )
    
    elif dataset_type == "2-operand-addition-verbal-v2":
        max_op = 500
        max_answer_value = 999
        operator = "+"
        template = "Question: What is the sum of {op1} and {op2}? Answer: "
        data = make_custom_prompts(
            model=model,
            num_prompts=n_data,
            max_op=max_op,
            max_answer_value=max_answer_value,
            operator=operator,
            template=template,
            correct_only=only_correct
        )
    
    elif dataset_type == "2-operand-addition-verbal-v3":
        max_op = 500
        max_answer_value = 999
        operator = "+"
        template = "Question: How much is {op1} plus {op2}? Answer: "
        data = make_custom_prompts(
            model=model,
            num_prompts=n_data,
            max_op=max_op,
            max_answer_value=max_answer_value,
            operator=operator,
            template=template,
            correct_only=only_correct
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if "verbal" in dataset_type:
        final_data = prepare_verbal_localization_data(data, model=model)
    else:
        final_data = prepare_localization_data(data, model=model)

    return final_data
