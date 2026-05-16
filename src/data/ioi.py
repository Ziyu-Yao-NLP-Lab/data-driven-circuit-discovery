import pdb
import re
import torch as t
import random

from src.data import ioi_dataset, ioi_dataset_filler, ioi_dataset_letter, ioi_dataset_passive, ioi_single_template
from src.utils.general_utils import is_correct
from src.utils import general_utils as gu

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

def prepare_localization_data(ds, model, tokenizer, dataset_type, n_data, only_correct=True):
    abc_dataset = ( 
        ds.gen_flipped_prompts(("S2", "RAND"))
    )
    final_data = []
    counter = 0
    for i in range(len(ds)):
        tmp = {}
        clean = ' '.join(ds.sentences[i].split()[:-1])
        corrupted = ' '.join(abc_dataset.sentences[i].split()[:-1])
        correct = tokenizer.encode(f' {ds.ioi_prompts[i]["IO"]}', add_special_tokens=False)[0] 
        incorrect = tokenizer.encode(f' {ds.ioi_prompts[i]["S"]}', add_special_tokens=False)[0] 

        if dataset_type == "passive":
            tmp1 = correct
            correct = incorrect
            incorrect = tmp1

        elif dataset_type == "letter":
            print("Using letter dataset, adjusting token indices.")
            correct = tokenizer.encode(f' {ds.ioi_prompts[i]["IO"]}', add_special_tokens=False)[1] 
            incorrect = tokenizer.encode(f' {ds.ioi_prompts[i]["S"]}', add_special_tokens=False)[1]
        
        # Skip examples where clean and corrupted have different token lengths
        clean_tok_len = len(tokenizer.encode(clean, add_special_tokens=False))
        corrupted_tok_len = len(tokenizer.encode(corrupted, add_special_tokens=False))
        if clean_tok_len != corrupted_tok_len:
            print(f"Number of positions must match, but do not: {clean_tok_len} (clean) != {corrupted_tok_len} (corrupted) [{clean}] [{corrupted}]. Skipping.")
            continue

        tmp['clean'] = clean
        tmp['corrupted'] = corrupted
        tmp['correct_idx'] = correct
        tmp['incorrect_idx'] = [incorrect]
        tmp['label'] = tokenizer.decode([correct]).strip()
        tmp['corrupted_labels'] = tokenizer.decode([incorrect]).strip()
        final_data.append(tmp)

        if only_correct:
            with t.no_grad():
                logits = model(tmp["clean"], return_type="logits")
            if is_correct(logits, tmp["correct_idx"], tmp["incorrect_idx"]) is False:
                print("Model got it wrong, skipping this sequence.")
                continue  # Skip this sequence if the model got it wrong
            counter += 1
        else:
            counter += 1
        
        if counter >= n_data:
            break  # Stop if we have enough correct examples
        
    return final_data


def create_data(model, tokenizer, dataset_type: str, n_data: int = 1000, only_correct: bool = False, return_dataloader: bool=False, prepend_bos: bool=False, random_seed: int=32):
    """Create n-gram induction data for the given model."""
    # Pin global random state so downstream random.shuffle and
    # create_n_person_data_from_original (random.sample/choice) are deterministic.
    random.seed(random_seed)
    if only_correct:
        n_data = n_data * 2  # Generate more data to account for filtering incorrect examples later

    if dataset_type == "mixed":
         ds_abba = ioi_dataset.IOIDataset("ABBA", N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)
         ds_baba = ioi_dataset.IOIDataset("BABA", N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)

    elif dataset_type == "ABBA" or dataset_type == "BABA":
        ds = ioi_dataset.IOIDataset(dataset_type, N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)
    elif dataset_type == "filler":
        ds = ioi_dataset_filler.IOIDataset("mixed", N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)
    elif dataset_type == "letter":
        ds = ioi_dataset_letter.IOIDataset("mixed", N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)
    elif dataset_type == "passive":
        ds = ioi_dataset_passive.IOIDataset("mixed", N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)
    elif dataset_type == "3-person":
        ds_abba = ioi_dataset.IOIDataset("ABBA", N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)
        ds_baba = ioi_dataset.IOIDataset("BABA", N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if only_correct:
        n_data = n_data // 2

    if dataset_type == "mixed":
        if return_dataloader:
            ds = ioi_dataset.IOIDataset("mixed", N=n_data, tokenizer=tokenizer, prepend_bos=prepend_bos, seed=random_seed)
            final_data = prepare_localization_data(ds, model, tokenizer, dataset_type, n_data, only_correct=only_correct)
        else:
            data_abba = prepare_localization_data(ds_abba, model, tokenizer, dataset_type, n_data//2, only_correct=only_correct)
            data_baba = prepare_localization_data(ds_baba, model, tokenizer, dataset_type, n_data//2, only_correct=only_correct)
            final_data = data_abba + data_baba
            random.shuffle(final_data)
    else:
        final_data = prepare_localization_data(ds, model, tokenizer, dataset_type, n_data, only_correct=only_correct)
    
    if dataset_type == "3-person":
        data_abba = prepare_localization_data(ds_abba, model, tokenizer, dataset_type, n_data//2, only_correct=only_correct)
        data_baba = prepare_localization_data(ds_baba, model, tokenizer, dataset_type, n_data//2, only_correct=only_correct)
        final_data = data_abba + data_baba
        io_first_data = create_n_person_data_from_original(model, tokenizer, data_abba, n_person=3, only_correct=only_correct, io_first=True)
        s_first_data = ioi.create_n_person_data_from_original(model, tokenizer, data_baba, n_person=3, only_correct=only_correct, io_first=False)
        final_data = io_first_data + s_first_data

    if return_dataloader:
        return final_data, ds
    return final_data

def create_n_person_data_from_original(model, tokenizer, data, n_person, only_correct, io_first):
    final_data = []
    counter = 0
    for each in data:
        clean = each['clean']
        clean_tok = clean.split()
        label = tokenizer.decode([each['correct_idx']]).strip()
        counterfactual_label = tokenizer.decode(each['incorrect_idx']).strip()
        names_in_inputs = [label, counterfactual_label]
        # Exclude names that are substrings of label/counterfactual_label or vice versa
        sampled_names = random.sample([
            name for name in NAMES
            if name not in names_in_inputs
            and not any(name in n or n in name for n in names_in_inputs)
        ], n_person)
        # make sure all sampled names are single token
        for i in range(len(sampled_names)):
            assert len(tokenizer.encode(f' {sampled_names[i]}', add_special_tokens=False)) == 1, f"Sampled name {sampled_names[i]} is not single token."

        if io_first:
            first_names_str = label + ' and ' + counterfactual_label
        else:
            first_names_str = counterfactual_label + ' and ' + label

        assert first_names_str in clean
        replace_first_names_str = ""
        for i, name in enumerate(sampled_names):
            if i == n_person - 1:
                replace_first_names_str += 'and ' + name
            else:
                replace_first_names_str += name + ', '

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
        
        # Use replace for the first substitution (unique multi-word phrase),
        # then regex word-boundary match for the second to avoid corrupting
        # names that contain counterfactual_label as a substring.
        new_clean = clean.replace(first_names_str, replace_first_names_str)
        new_clean = re.sub(r'\b' + re.escape(counterfactual_label) + r'\b', replace_second_names_str, new_clean)

        all_used_names = sampled_names + [label, new_label, counterfactual_label]
        sample_counterfactual_names = random.sample([
            name for name in NAMES
            if name not in all_used_names
            and not any(name in n or n in name for n in all_used_names)
        ], n_person-1)

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
        new_corrupted = re.sub(re.escape(replace_second_names_str), replace_second_cnames_str, new_clean, count=1)
        # Skip examples where clean and corrupted have different token lengths
        clean_tok_len = len(tokenizer.encode(new_clean, add_special_tokens=False))
        corrupted_tok_len = len(tokenizer.encode(new_corrupted, add_special_tokens=False))
        if clean_tok_len != corrupted_tok_len:
            print(f"Token length mismatch in 3-person data: {clean_tok_len} (clean) != {corrupted_tok_len} (corrupted). Skipping.")
            continue

        tmp = {
            'clean': new_clean,
            'corrupted': new_corrupted,
            'correct_idx': tokenizer.encode(f' {new_label}', add_special_tokens=False)[0],
            'incorrect_idx': [tokenizer.encode(f' {name}', add_special_tokens=False)[0] for name in sampled_names],
            'label': new_label,
            'corrupted_labels': sampled_names
        }

        if only_correct:
            with t.no_grad():
                logits = model(tmp["clean"], return_type="logits")
            
            if gu.is_correct(logits, tmp["correct_idx"], tmp["incorrect_idx"]) is False:
                print("Model got it wrong, skipping this sequence.")
                continue  # Skip this sequence if the model got it wrong

        final_data.append(tmp)

    return final_data