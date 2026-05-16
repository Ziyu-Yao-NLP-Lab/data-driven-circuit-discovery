import pdb
import pandas as pd
import os
from transformers import AutoTokenizer
import random

from src.utils import general_utils as gu
from src.data import ioi_dataset, ioi_dataset_filler, ioi_dataset_passive, ioi_dataset_colon, ioi_letter_dataset

NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

def create_dataset(tokenizer, dataset_type, n_samples, save_path=None):    
    if dataset_type == "filler":
        ds = ioi_dataset_filler.IOIDataset("mixed", N=n_samples, tokenizer=tokenizer)
    elif dataset_type == "passive":
        ds = ioi_dataset_passive.IOIDataset("mixed", N=n_samples, tokenizer=tokenizer)
    elif dataset_type == "letters":
        ds = ioi_letter_dataset.IOIDataset("mixed", N=n_samples, tokenizer=tokenizer)
    elif dataset_type == "colon":
        ds = ioi_dataset_colon.IOIDataset("mixed", N=n_samples, tokenizer=tokenizer)
    elif dataset_type == "mixed" or "ABBA" or "BABA":
        ds = ioi_dataset.IOIDataset(dataset_type, N=n_samples, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    abc_dataset = (  # TODO seeded
        ds.gen_flipped_prompts(("S2", "RAND"))
    )
    # abc_dataset2 = (  # TODO seeded
    #     ds.gen_flipped_prompts(("IO", "RAND"))
    # )
    # abb_dataset = (  # TODO seeded
    #     ds.gen_flipped_prompts(("S2", "IO"))
    # )

    # d = {'clean': [], 'corrupted': [], 'corrupted_hard': [], 'correct_idx': [], 'incorrect_idx': []}
    d = {'clean': [], 'corrupted': [], 'correct_idx': [], 'incorrect_idx': [], 'label': '', 'corrupted_labels': []}
    for i in range(len(ds)):
        clean = ' '.join(ds.sentences[i].split()[:-1])
        corrupted = ' '.join(abc_dataset.sentences[i].split()[:-1])
        # corrupted_hard = ' '.join(abb_dataset.sentences[i].split()[:-1])
        correct = tokenizer.encode(f' {ds.ioi_prompts[i]["IO"]}', add_special_tokens=False)[0] 
        #ds.toks[i, ds.word_idx['IO'][i]].item()
        incorrect = tokenizer.encode(f' {ds.ioi_prompts[i]["S"]}', add_special_tokens=False)[0] 
        #incorrect = ds.toks[i, ds.word_idx['S'][i]].item()

        if dataset_type == "passive":
            tmp = correct
            correct = incorrect
            incorrect = tmp

        elif dataset_type == "letters":
            print("Using letter dataset, adjusting token indices.")
            correct = tokenizer.encode(f' {ds.ioi_prompts[i]["IO"]}', add_special_tokens=False)[1] 
            incorrect = tokenizer.encode(f' {ds.ioi_prompts[i]["S"]}', add_special_tokens=False)[1]
        
        d['clean'].append(clean)
        d['corrupted'].append(corrupted)
        # d['corrupted_hard'].append(corrupted_hard)
        d['correct_idx'].append(correct)
        d['incorrect_idx'].append([incorrect])
        d['label'] = tokenizer.decode([correct]).strip()
        d['corrupted_labels'].append(tokenizer.decode([incorrect]).strip())

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame.from_dict(d)
        df = df.sample(frac=1)
        df.to_csv(f'{save_path}', index=False)
    
    return d


def create_n_person_data_from_original(tokenizer, each, n_person):
    clean = each['clean']
    label = tokenizer.decode([each['correct_idx']]).strip()
    clean_toks = clean.split()
    clean_names = [clean_toks[1], clean_toks[3]]
    sampled_names = random.sample([name for name in NAMES if name not in clean_names], n_person)
    # make sure all sampled names are single token
    for i in range(len(sampled_names)):
        assert len(tokenizer.encode(f' {sampled_names[i]}', add_special_tokens=False)) == 1, f"Sampled name {sampled_names[i]} is not single token."
    first_names_str = clean_names[0] + ' and ' + clean_names[1]
    replace_first_names_str = ""
    len_sampled_names = len(sampled_names)
    for i, name in enumerate(sampled_names):
        if i == len_sampled_names - 1:
            replace_first_names_str += 'and ' + name
        else:
            replace_first_names_str += name + ', '
    clean_names.remove(label)
    new_label = random.choice(sampled_names)
    sampled_names.remove(new_label)
    replace_second_names_str = ""
    len_second_names = len(sampled_names)
    for i, name in enumerate(sampled_names):
        if i == len_second_names - 1:
            replace_second_names_str += 'and ' + name
        elif len_second_names == 2 and i == 0:
            replace_second_names_str += name + ' '
        else:
            replace_second_names_str += name + ', '
    new_clean = clean.replace(first_names_str, replace_first_names_str).replace(clean_names[0], replace_second_names_str)
    sample_counterfactual_names = random.sample([name for name in NAMES if name not in sampled_names+[label]], n_person-1)
    corrupted = each['corrupted']
    replace_second_cnames_str = ""
    len_second_names = len(sample_counterfactual_names)
    for i, name in enumerate(sample_counterfactual_names):
        if i == len_second_names - 1:
            replace_second_cnames_str += 'and ' + name
        elif len_second_names == 2 and i == 0:
            replace_second_cnames_str += name + ' '
        else:
            replace_second_cnames_str += name + ', '
    new_corrupted = new_clean.replace(replace_second_names_str, replace_second_cnames_str)
    # corrupted_label = random.sample(sampled_names, 1)[0]
    return {
        'clean': new_clean,
        'corrupted': new_corrupted,
        'correct_idx': tokenizer.encode(f' {new_label}', add_special_tokens=False)[0],
        'incorrect_idx': [tokenizer.encode(f' {name}', add_special_tokens=False)[0] for name in sampled_names],
        'label': new_label,
        'corrupted_labels': sampled_names
    }