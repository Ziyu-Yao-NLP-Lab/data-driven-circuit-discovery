import torch as t
from jaxtyping import Int
from transformer_lens import HookedTransformer
import pdb
import pandas as pd
import os

# 1. Setup Names List
NAMES = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua", "Amanda", "Daniel", "David",
    "James", "Robert", "John", "Joseph", "Andrew", "Ryan", "Brandon", "Jason", "Justin", "Sarah",
    "William", "Jonathan", "Stephanie", "Brian", "Nicole", "Nicholas", "Anthony", "Heather", "Eric", "Elizabeth",
    "Adam", "Megan", "Melissa", "Kevin", "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel",
    "Laura", "Lauren", "Amber", "Brittany", "Danielle", "Richard", "Kimberly", "Jeffrey", "Amy", "Crystal",
    "Michelle", "Tiffany", "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles", "Rebecca", "Jacob",
    "Stephen", "Patrick", "Sean", "Erin", "Jamie", "Kelly", "Samantha", "Nathan", "Sara", "Dustin",
    "Paul", "Angela", "Tyler", "Scott", "Katherine", "Andrea", "Gregory", "Erica", "Mary", "Travis",
    "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen", "Jose", "Alexander", "Jesse", "Katie", "Lindsay",
    "Shannon", "Vanessa", "Courtney", "Christine", "Alicia", "Cody", "Allison", "Bradley", "Samuel",
]


def generate_repeated_tokens(
    tokenizer, seq_len: int, batch: int = 1
):
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    
    prefix = (t.ones(batch, 1) * tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, tokenizer.vocab_size, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1)
    return rep_tokens


def create_datasets(tokenizer, n_samples=500, seq_len=6, type = None, save_path=None):
    # create original
    original_data = []
    for _ in range(n_samples):
        rep_tokens = generate_repeated_tokens(tokenizer, seq_len=seq_len)
        # d['clean'].append(clean)
        # d['corrupted'].append(corrupted)
        # # d['corrupted_hard'].append(corrupted_hard)
        # d['correct_idx'].append(correct)
        # d['incorrect_idx'].append([incorrect])
        # d['label'] = tokenizer.decode([correct]).strip()
        # d['corrupted_labels'].append(tokenizer.decode([incorrect]).strip())
        pdb.set_trace()
        original_data.append({
            "clean": tokenizer.decode(rep_tokens[0]),
            "target_ids": rep_tokens[0, 1 + seq_len :],
        })
        
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)



