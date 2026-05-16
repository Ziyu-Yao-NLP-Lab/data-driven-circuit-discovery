import os
import pdb
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
import torch
import gc
import time
import transformer_lens as lens
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from transformer_lens.utils import get_attention_mask

def save_data_to_csv(data, save_path: str):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    df = pd.DataFrame.from_dict(data)
    df.to_csv(f'{save_path}', index=False)

def is_correct(logits, correct_token_idx, incorrect_tokens_idx):
    if logits.ndim == 3:
        logits = logits[0]
    if logits.ndim == 2:
        final_logits = logits[-1, :]
    if logits.ndim ==1:
        final_logits = logits
    correct_logit = final_logits[correct_token_idx]
    for inc_idx in incorrect_tokens_idx:
        incorrect_logit = final_logits[inc_idx]
        if incorrect_logit >= correct_logit:
            return False
    return True

    
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def save_file(data, file_name):
    """Save data."""
    with open(file_name, 'w') as f:
        f.write(data)

def read_file(file_name):
    """Read plain text data."""
    with open(file_name, 'r') as f:
        return f.read()

def save_json_file(data, file_name):
    """Save data to a JSON file."""
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)
    
def read_csv(file_path: str) -> pd.DataFrame:
    """Read CSV file and return as a pandas DataFrame."""
    return pd.read_csv(file_path)


def model2family(model_name: str):
    if 'gpt2' in model_name.lower():
        return 'gpt2'
    elif 'pythia' in model_name.lower():
        return 'pythia'
    elif "llama-3.1-8b-instruct" in model_name.lower():
        return 'llama-3.1-8b-instruct'
    elif "llama-3.1-8b" in model_name.lower():
        return "llama-3.1-8b"
    elif 'llama-3-8b' in model_name.lower():
        return 'llama-3-8b'
    elif "qwen2.5-7b-instruct" in model_name.lower():
        return 'qwen2.5-7b-instruct'
    elif 'qwen2.5-7b' in model_name.lower():
        return 'qwen2.5-7b'
    elif "qwen2.5-0.5b-instruct" in model_name.lower():
        return 'qwen2.5-0.5b-instruct'
    elif "qwen2.5-0.5b" in model_name.lower():
        return 'qwen2.5-0.5b'
    elif 'qwen3-8b' in model_name.lower():
        return 'qwen3-8b'
    elif 'gemma-2-9b' in model_name.lower():
        return 'gemma-2-9b'
    elif 'pythia-6.9b' in model_name.lower():
        return 'pythia-6.9b'
    elif 'mistral' in model_name.lower():
        return 'mistral'
    elif 'olmo' in model_name.lower():
        return 'olmo'
    else:
        raise ValueError(f"Couldn't find model family for model: {model_name}")


def get_act_name(act: str, layer: int) -> str:
    """
    Returns the full hook name for a given activation and layer.

    Common activations:
    - 'z'   -> attention head output
    - 'q'   -> query
    - 'k'   -> key
    - 'v'   -> value
    - 'attn_out' -> attention output (post-head-mix)
    - 'mlp_out'  -> MLP output
    """
    return f"blocks.{layer}.attn.hook_{act}"

def get_logit_positions(logits: torch.Tensor, token_idx):
    # Always focus on the last token in the sequence
    last_logits = logits[-1] if logits.ndim == 2 else logits[:, -1, :]
    
    sorted_idx = torch.argsort(last_logits, dim=-1, descending=True)

    if last_logits.ndim == 1:
        # Single sequence
        pos = (sorted_idx == token_idx).nonzero(as_tuple=True)[0].item()
    else:
        # Batch of sequences
        pos = (sorted_idx == token_idx).nonzero(as_tuple=False)[:, 1]
    return pos


def tokenize_plus(model, inputs, max_length = None):
    """
    Tokenizes the input strings using the provided model.

    Args:
        model (HookedTransformer): The model used for tokenization.
        inputs (List[str]): The list of input strings to be tokenized.

    Returns:
        tuple: A tuple containing the following elements:
            - tokens (torch.Tensor): The tokenized inputs.
            - attention_mask (torch.Tensor): The attention mask for the tokenized inputs.
            - input_lengths (torch.Tensor): The lengths of the tokenized inputs.
            - n_pos (int): The maximum sequence length of the tokenized inputs.
    """
    if max_length is not None:
        old_n_ctx = model.cfg.n_ctx
        model.cfg.n_ctx = max_length
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
    if max_length is not None:
        model.cfg.n_ctx = old_n_ctx
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos

class EAPCounterfactualDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # return row['corrupted'], row['clean'], [row['correct_idx'], row['incorrect_idx']]
        return row['corrupted'], row['clean'], [row['correct_idx']] + json.loads(row['incorrect_idx'])
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    try:
        labels = torch.tensor(labels)
    except:
        labels = [torch.tensor(lbl) for lbl in labels]
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return clean, corrupted, labels


class EAPDataset_from_data(Dataset):
    def __init__(self, data):
        self.df = data

    def __len__(self):
        return len(self.df)
    

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
        return row['clean'], row['corrupted'], [row['correct_idx']] + json.loads(row['incorrect_idx'])
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)

class EAPDataset(Dataset):
    def __init__(self, filepath, n_samples=None):
        self.df = pd.read_csv(filepath)
        if n_samples:
            self.df = self.df.head(n_samples)

    def __len__(self):
        return len(self.df)
    

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
        return row['clean'], row['corrupted'], [row['correct_idx']] + json.loads(row['incorrect_idx'])
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_model_name(obj):
    """
    Return the name or path of a loaded model or tokenizer.
    Works for TransformerLens models and Hugging Face models/tokenizers.
    """
    # TransformerLens model
    if hasattr(obj, "cfg") and hasattr(obj.cfg, "model_name"):
        return obj.cfg.model_name

    # Hugging Face model or tokenizer (AutoModel, AutoTokenizer, etc.)
    if hasattr(obj, "name_or_path"):
        return obj.name_or_path

    # TransformerLens model wrapping a Hugging Face model
    if hasattr(obj, "hf_model") and hasattr(obj.hf_model, "name_or_path"):
        return obj.hf_model.name_or_path

    raise ValueError("Could not determine model name — object has no recognizable name attribute.")

def load_model(model_name: str, cache_dir: str):
    """
    Function to load the model

    Arguments:
    model_name (str): name of the model (short alias or HuggingFace ID)
    cache_dir (str): directory to store/load cache of model. Passed through to
        TransformerLens / HuggingFace from_pretrained. Required.
    """
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if "llama" in model_name.lower():
        if "llama-3.1-8b-instruct" in model_name.lower():
            model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif "llama-3.1-8b" in model_name.lower():
            model_name = "meta-llama/Llama-3.1-8B"
        elif "llama-3-8b" in model_name.lower():
            model_name = "meta-llama/Llama-3-8B"
        else:
            raise ValueError(f"Unknown Llama model: {model_name}")
        
        model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
            center_unembed=False,
            cache_dir=cache_dir,
            fold_ln=False,
            device='cuda',
            dtype=torch.float16
        )
        tokenizer = model.tokenizer
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True

    elif "Qwen" in model_name or "pythia" in model_name.lower():
        inner_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = lens.HookedTransformer.from_pretrained(
            model_name=model_name, 
            hf_model=inner_model, 
            tokenizer=tokenizer,
            fold_ln=True, 
            center_unembed=True, 
            center_writing_weights=False, 
            device="cuda",
            dtype="float16",
        )
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True

    else:
        model = lens.HookedTransformer.from_pretrained(
            model_name,
            dtype=torch.float32,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=True,
            cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.use_attn_result = True

    return model, tokenizer