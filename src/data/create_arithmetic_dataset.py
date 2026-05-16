"""
This file contains functions for generating arithmetic data.
"""

import os
import pickle
import random
import smtplib
from email.mime.text import MIMEText
from typing import List, Tuple
import pdb

import torch
import transformer_lens as lens
from tqdm import tqdm


def create_two_operand_prompts(
    tokenizer,
    num_prompts: int,
    max_op: int,
    max_answer_value: int,
    operator: str,
    correct_only: bool = False,
    random_seed: int = 11,
    filter_type = None
):
    """
    Generates two-operand prompts (e.g., "A+B=") and their corresponding
    answers.
    """
    max_single_digit_token = max_op if max_op < max_answer_value else max_answer_value
    single_digit_tokens = [str(i) for i in range(max_single_digit_token) if len(tokenizer.encode(f'{i}', add_special_tokens=False)) == 1] 
    print(f"len single_digit_tokens: {len(single_digit_tokens)}")

    expressions = []

    for op1 in tqdm(range(1, max_op), desc="Generating prompts", leave=False):
        for op2 in range(1, max_op):
            if str(op1) not in single_digit_tokens or str(op2) not in single_digit_tokens: # skip if both op1 and op2 are not single-digit tokens
                continue

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
            
            if result not in single_digit_tokens and result <= max_answer_value:
                if filter_type == "even" and result % 2 != 0:
                    continue
                elif filter_type == "odd" and result % 2 == 0:
                    continue
                elif filter_type == "greater_than_500" and result <= 500:
                    continue
                expressions.append(prompt)
            else:
                print(f"Skipping prompt: {prompt} with result: {result}")
    if not expressions:
        raise ValueError("No valid expressions generated.")
    
    random.seed(random_seed)
    random.shuffle(expressions)

    pbar = tqdm(total=num_prompts, desc="Creating counterfactual prompts")
    data = []
    if not correct_only:
        # Just take the first `num_prompts` and get their answers
        for idx, prompt in enumerate(expressions[:num_prompts]):
            tmp = {}
            tmp['clean'] = prompt
            try:
                tmp['corrupted'] = expressions[idx+1]  # just use the next expression as corrupted version
            except:
                tmp['corrupted'] = expressions[idx-1]  # wrap around if at the end
            
            tmp['correct_idx'] = tokenizer.encode(str(eval(prompt.rstrip("="))), add_special_tokens=False)[0]

            incorrect_label = str(eval(tmp['corrupted'].rstrip("=")))
            tmp['incorrect_idx'] = [tokenizer.encode(incorrect_label, add_special_tokens=False)[0]]

            tmp['label'] = tokenizer.decode([tmp['correct_idx']]).strip()
            tmp['corrupted_labels'] = [tokenizer.decode([tmp['incorrect_idx'][0]]).strip()]
            data.append(tmp)
            pbar.update(1)
            
        pbar.close()
        return data

def make_three_operand_prompts_prev(
    tokenizer,
    num_prompts: int,
    max_op: int,
    max_answer_value: int,
    template: str,
    correct_only: bool = True,
    random_seed: int = 11
):
    """
    Generates three-operand prompts (e.g., "A+B-C=") and their corresponding
    answers, optionally filtering for those the model answers correctly.
    """
    random.seed(random_seed)
    expressions = []
    data = []

    max_single_digit_token = max_op if max_op < max_answer_value else max_answer_value
    single_digit_tokens = [str(i) for i in range(max_single_digit_token) if len(tokenizer.encode(f'{i}', add_special_tokens=False)) == 1] 
    print(f"len single_digit_tokens: {len(single_digit_tokens)}")
    
    # Extract operators from template
    parts = template.replace("=","").strip().split(" ")
    op1_char = parts[1]
    op2_char = parts[3]

    for opA in tqdm(range(1, max_op), desc="Generating prompts", leave=False):
        for opB in range(1, max_op):
            for opC in range(1, max_op):
                if (str(opA) not in single_digit_tokens or 
                    str(opB) not in single_digit_tokens or 
                    str(opC) not in single_digit_tokens):
                    continue  # skip if any operand is not a single-digit token
                expr_str = f"{opA} {op1_char} {opB} {op2_char} {opC}"
                result = eval(expr_str)
                
                if 0 <= result < max_answer_value and str(result) not in single_digit_tokens:
                    prompt = template.format(A=opA, B=opB, C=opC)
                    expressions.append(prompt)

    if not expressions:
        raise ValueError("No valid expressions generated.")

    random.shuffle(expressions)

    pbar = tqdm(total=num_prompts, desc="Creating counterfactual prompts")
    if not correct_only:
        for idx, prompt in enumerate(expressions[:num_prompts]):
            tmp = {}
            tmp['clean'] = prompt
            try:
                tmp['corrupted'] = expressions[idx+1]  # just use the next expression as corrupted version
            except:
                tmp['corrupted'] = expressions[idx-1]  # wrap around if at the end
            
            tmp['correct_idx'] = tokenizer.encode(str(eval(prompt.rstrip("="))), add_special_tokens=False)[0]

            incorrect_label = str(eval(tmp['corrupted'].rstrip("=")))
            tmp['incorrect_idx'] = [tokenizer.encode(incorrect_label, add_special_tokens=False)[0]]

            tmp['label'] = tokenizer.decode([tmp['correct_idx']]).strip()
            tmp['corrupted_labels'] = [tokenizer.decode([tmp['incorrect_idx'][0]]).strip()]
            data.append(tmp)
            pbar.update(1)
            
        pbar.close()
        return data

    
