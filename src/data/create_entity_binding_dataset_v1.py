import pandas as pd
import os
from transformers import AutoTokenizer
import random
import pdb

# --- Constants ---
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

HOUSEHOLD_ITEMS = [
    "egg", "fan", "tea", "engine", "plate", "gift", "wire", "watch", "cross", "boat",
    "game", "rose", "shell", "seed", "magnet", "suit", "ticket", "glass", "tie", "card",
    "brain", "fig", "wheel", "machine", "note", "drink", "bread", "camera", "bill",
    "chemical", "clock", "flower", "creature", "rock", "plant", "sheet", "leaf", "block",
    "newspaper", "disk", "boot", "medicine", "coffee", "book", "ball", "string", "fish",
    "crown", "branch", "phone", "plane", "apple", "cup", "bell", "brick", "document",
    "file", "bus", "bag", "drug", "pot", "computer", "mirror", "stone", "radio", "dress",
    "meat", "train", "bomb", "letter", "guitar", "hat", "map", "magazine", "coat",
    "television", "painting", "picture", "milk", "pipe", "ice", "key",
]

COUNTRIES = [
    'Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'Egypt', 'France', 'Georgia', 
    'Germany', 'India', 'Iran', 'Iraq', 'Israel', 'Italy', 'Japan', 'Jordan', 'Mexico', 
    'Montserrat', 'Pakistan', 'Russia', 'Scotland', 'Singapore', 'Spain', 'Sweden', 'Turkey'
]

NAMES = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua", "Amanda",
    "Daniel", "David", "James", "Robert", "John", "Joseph", "Andrew", "Ryan", "Brandon",
    "Jason", "Justin", "Sarah", "William", "Jonathan", "Stephanie", "Brian", "Nicole",
    "Nicholas", "Anthony", "Heather", "Eric", "Elizabeth", "Adam", "Megan", "Melissa",
    "Kevin", "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel", "Laura", "Lauren",
    "Amber", "Brittany", "Danielle", "Richard", "Kimberly", "Jeffrey", "Amy", "Crystal",
    "Michelle", "Tiffany", "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles",
    "Rebecca", "Jacob", "Stephen", "Patrick", "Sean", "Erin", "Jamie", "Kelly", "Samantha",
    "Nathan", "Sara", "Dustin", "Paul", "Angela", "Tyler", "Scott", "Katherine", "Andrea",
    "Gregory", "Erica", "Mary", "Travis", "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen",
    "Jose", "Alexander", "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney",
    "Christine", "Alicia", "Cody", "Allison", "Bradley", "Samuel",
]

# --- Helpers ---

def filter_single_tokens(tokenizer, word_list, prepend_space=False):
    """Filters a list to keep only words that tokenize to a single token."""
    valid_words = []
    for word in word_list:
        text = " " + word if prepend_space else word
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 1:
            valid_words.append(word)
    return valid_words

def save_to_csv(data, path):
    """
    Saves list of dicts to CSV, STRICTLY enforcing the required columns.
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    df = pd.DataFrame(data)
    
    # STRICTLY FORCE COLUMNS
    required_cols = ['clean', 'corrupted', 'correct_idx', 'incorrect_idx', 'label', 'corrupted_labels']
    df = df[required_cols]
    
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

# --- Sample Generators ---

def get_inverse_box_sample(tokenizer, n, sampled_items, sampled_letters):
    """
    Format: "Which box is the {Item} in?" -> Label: "{Letter}"
    """
    prompt = ""
    for i, (item, letter) in enumerate(zip(sampled_items, sampled_letters)):
        prompt += f"The {item} is in box {letter}. "
    
    selected_index = random.randint(0, n - 1)
    target_item = sampled_items[selected_index]
    target_letter = sampled_letters[selected_index]
    
    # Clean: Which box is the medicine in?
    clean_prompt = prompt + f"Which box is the {target_item} in?"
    
    # Corrupted: Ask about a different item in the context
    available_indices = [i for i in range(n) if i != selected_index]
    counterfactual_idx = random.choice(available_indices)
    counterfactual_item = sampled_items[counterfactual_idx]
    counterfactual_prompt = prompt + f"Which box is the {counterfactual_item} in?"
    
    # Label is the LETTER
    label_token = " " + target_letter
    
    # Corrupted Labels: Other letters in context
    corrupted_labels = [" " + sampled_letters[idx] for idx in range(n) if idx != selected_index]

    correct_idx = tokenizer.encode(label_token, add_special_tokens=False)
    incorrect_idx = [tokenizer.encode(cl, add_special_tokens=False)[0] for cl in corrupted_labels]
    
    assert len(correct_idx) == 1, f"Tokenization error: {correct_idx}"

    return {
        "clean": clean_prompt,
        "corrupted": counterfactual_prompt,
        "correct_idx": correct_idx[0],
        "incorrect_idx": incorrect_idx,
        "label": target_letter,
        "corrupted_labels": [letter for idx, letter in enumerate(sampled_letters) if idx != selected_index]
    }

def get_inverse_country_sample(tokenizer, n, sampled_entity, sampled_items, all_items, all_entities, dataset_type="country"):
    """
    Format: "Who lives in {Country}?" -> Label: "{Person}"
    """
    
    # 1. Build Context
    context_str = ""
    for item, entity in zip(sampled_items, sampled_entity):
        context_str += f"{entity} lives in {item}. "
    
    # 2. Select Targets
    selected_index = random.randint(0, n - 1)
    target_country = sampled_items[selected_index] 
    target_person = sampled_entity[selected_index] 

    # 3. Handle Fillers
    distractor_pool = [item for item in all_items if item != target_country]
    distractor_item = random.choice(distractor_pool)
    
    if dataset_type == "country-filler-related":
        # {Person} lives in {Country} and works in {Distractor}.
        filler_sentence = f"{target_person} lives in {target_country} and works in {distractor_item}. "
        context_str = context_str.replace(f"{target_person} lives in {target_country}. ", filler_sentence)
        
    elif dataset_type == "country-filler-unrelated":
        # {Other_Person} lives in {Other_Country} and works in {Distractor}.
        unrelated_idx = selected_index + 1 if selected_index < n - 1 else selected_index - 1
        unrelated_entity = sampled_entity[unrelated_idx]
        unrelated_label = sampled_items[unrelated_idx]
        
        filler_sentence = f"{unrelated_entity} lives in {unrelated_label} and works in {distractor_item}. "
        context_str = context_str.replace(f"{unrelated_entity} lives in {unrelated_label}. ", filler_sentence)

    # 4. Build Prompts
    clean_prompt = f"{context_str}Who lives in {target_country}?"
    
    # Corrupted: Ask about a different country
    counterfactual_index = random.randint(0, n - 1)
    while counterfactual_index == selected_index:
        counterfactual_index = random.randint(0, n - 1)
    
    counterfactual_country = sampled_items[counterfactual_index]
    counterfactual_prompt = f"{context_str}Who lives in {counterfactual_country}?"
    
    # 5. Labels
    label_token = " " + target_person
    corrupted_labels = [" " + sampled_entity[idx] for idx in range(n) if idx != selected_index]

    correct_idx = tokenizer.encode(label_token, add_special_tokens=False)
    incorrect_idx = [tokenizer.encode(cl, add_special_tokens=False)[0] for cl in corrupted_labels]
    
    assert len(correct_idx) == 1, f"Tokenization error: {correct_idx}"

    return {
        "clean": clean_prompt, 
        "corrupted": counterfactual_prompt, 
        "correct_idx": correct_idx[0], 
        "incorrect_idx": incorrect_idx, 
        "label": target_person, 
        "corrupted_labels": corrupted_labels
    }

def get_inverse_fixed_position_sample(tokenizer, n, sampled_entities, sampled_items, all_entities, target_pos_index):
    """
    Format: "Who lives in {Country}?" (Fixed Position)
    """
    context_str = ""
    for item, entity in zip(sampled_items, sampled_entities):
        context_str += f"{entity} lives in {item}. "
    
    selected_index = target_pos_index
    target_country = sampled_items[selected_index]
    target_person = sampled_entities[selected_index]

    clean_prompt = f"{context_str}Who lives in {target_country}?"
    
    # Corrupted: Ask about a random country not in context
    available_countries = [c for c in COUNTRIES if c not in sampled_items]
    if not available_countries: available_countries = sampled_items 
    counterfactual_country = random.choice(available_countries)
    counterfactual_prompt = f"{context_str}Who lives in {counterfactual_country}?"
    
    label_token = " " + target_person
    corrupted_labels = [" " + sampled_entities[idx] for idx in range(n) if idx != selected_index]

    correct_idx = tokenizer.encode(label_token, add_special_tokens=False)
    incorrect_idx = [tokenizer.encode(cl, add_special_tokens=False)[0] for cl in corrupted_labels]
    
    assert len(correct_idx) == 1, f"Tokenization error: {correct_idx}"

    return {
        "clean": clean_prompt, 
        "corrupted": counterfactual_prompt, 
        "correct_idx": correct_idx[0], 
        "incorrect_idx": incorrect_idx, 
        "label": target_person, 
        "corrupted_labels": corrupted_labels
    }

def get_box_data_comma(tokenizer, n, sampled_items, sampled_letters):
    prompt = ""
    i = 0
    for item, letter in zip(sampled_items, sampled_letters):
        i += 1
        if i == 1:
            prompt += f"The {item} is in box {letter}, "
        elif i == n:
            prompt += f"the {item} is in box {letter}. "
        else:
            prompt += f"the {item} is in box {letter}, "
    
    # randomly select a number between 0 and n-1
    selected_index = random.randint(0, n - 1)
    counterfactual_index = random.randint(0, n - 1)
    while counterfactual_index == selected_index:
        counterfactual_index = random.randint(0, n - 1)
    assert counterfactual_index != selected_index
    clean_prompt = prompt + f"Box {sampled_letters[selected_index]} contains the"
    # sample a letter that is not in sampled_letters
    counterfactual_letter = random.choice([letter for letter in LETTERS if letter not in sampled_letters])
    counterfactual_prompt = prompt + f"Box {counterfactual_letter} contains the"
    label = " " + sampled_items[selected_index]
    # corrupted_label = " " + sampled_items[counterfactual_index]
    corrupted_labels = [" " + sampled_items[idx] for idx in range(n) if idx != selected_index]

    correct_idx = tokenizer.encode(label, add_special_tokens=False)
    incorrect_idx = [tokenizer.encode(corrupted_label, add_special_tokens=False)[0] for corrupted_label in corrupted_labels]
    assert len(correct_idx) == 1, f"Tokenization error: {correct_idx}, {incorrect_idx}"

    return {"clean": clean_prompt, "corrupted": counterfactual_prompt, "correct_idx": correct_idx[0], "incorrect_idx": incorrect_idx, "label": sampled_items[selected_index], "corrupted_labels": [item for idx, item in enumerate(sampled_items) if idx != selected_index]}

def get_box_sample(tokenizer, n, sampled_items, sampled_letters):
    prompt = ""
    for i, (item, letter) in enumerate(zip(sampled_items, sampled_letters)):
        prompt += f"The {item} is in box {letter}. "
    
    # Select Target
    selected_index = random.randint(0, n - 1)
    
    # Build Prompts
    clean_prompt = prompt + f"Box {sampled_letters[selected_index]} contains the"
    
    # Counterfactual (Targeting a different letter)
    available_letters = [l for l in LETTERS if l not in sampled_letters]
    counterfactual_letter = random.choice(available_letters)
    counterfactual_prompt = prompt + f"Box {counterfactual_letter} contains the"
    
    # Labels
    label_word = sampled_items[selected_index]
    label_token = " " + label_word
    
    # Corrupted labels (all other items in context)
    corrupted_labels = [" " + sampled_items[idx] for idx in range(n) if idx != selected_index]

    # Token Indices
    correct_idx = tokenizer.encode(label_token, add_special_tokens=False)
    incorrect_idx = [tokenizer.encode(cl, add_special_tokens=False)[0] for cl in corrupted_labels]
    
    assert len(correct_idx) == 1, f"Tokenization error: {correct_idx}"
    

    return {
        "clean": clean_prompt,
        "corrupted": counterfactual_prompt,
        "correct_idx": correct_idx[0],
        "incorrect_idx": incorrect_idx,
        "label": label_word,
        "corrupted_labels": [item for idx, item in enumerate(sampled_items) if idx != selected_index]
    }

# --- Dataset Creators ---

def create_box_dataset_base(tokenizer, n_samples, data_type, n_entities=10, random_seed=11, save_path="data/box"):
    random.seed(random_seed)
    
    # Filter Vocab once
    valid_items = filter_single_tokens(tokenizer, HOUSEHOLD_ITEMS)
    valid_letters = filter_single_tokens(tokenizer, LETTERS)
    
    all_aggregated_data = []

    for n in range(2, n_entities + 1):
        data = []
        # Attempt to generate n_samples unique samples
        attempts = 0
        while len(data) < n_samples and attempts < n_samples * 5:
            attempts += 1
            sampled_items = random.sample(valid_items, n)
            sampled_letters = random.sample(valid_letters, n)
            
            if data_type == "comma":
                sample = get_box_data_comma(tokenizer, n, sampled_items, sampled_letters)
            elif data_type == "period":
                sample = get_box_sample(tokenizer, n, sampled_items, sampled_letters)
            else:
                raise ValueError(f"Unknown type: {data_type}")
            
            # Simple deduplication check (using clean prompt as key)
            if not any(d['clean'] == sample['clean'] for d in data):
                data.append(sample)
        
        all_aggregated_data.extend(data)
        
        # Save specific N file
        if data_type == "comma":
            file_path = f"{save_path}/box_{n}_entities_comma.csv"
        elif data_type == "period":
            file_path = f"{save_path}/box_{n}_entities_period.csv"
        else:
            raise ValueError(f"Unknown type: {data_type}")
        save_to_csv(data, file_path)

    # Save aggregated file (shuffled subset)
    random.shuffle(all_aggregated_data)
    final_subset = all_aggregated_data[:n_samples]
    if data_type == "comma":
        save_to_csv(final_subset, f"{save_path}/box_all_entities_comma.csv")
    elif data_type == "period":
        save_to_csv(final_subset, f"{save_path}/box_all_entities_period.csv")
    else:
        raise ValueError(f"Unknown type: {data_type}")


def create_instruct_box_dataset(tokenizer, n_samples, n_entities=10, random_seed=11, save_path="data/instruct_box"):
    random.seed(random_seed)
    valid_items = filter_single_tokens(tokenizer, HOUSEHOLD_ITEMS)
    valid_letters = filter_single_tokens(tokenizer, LETTERS)
    
    all_aggregated_data = []

    for n in range(2, n_entities + 1):
        data = []
        attempts = 0
        while len(data) < n_samples and attempts < n_samples * 5:
            attempts += 1
            sampled_items = random.sample(valid_items, n)
            sampled_letters = random.sample(valid_letters, n)
            
            sample = get_inverse_box_sample(tokenizer, n, sampled_items, sampled_letters)
            
            if not any(d['clean'] == sample['clean'] for d in data):
                data.append(sample)
        
        all_aggregated_data.extend(data)
        file_path = f"{save_path}/box_{n}_entities_instruct.csv"
        save_to_csv(data, file_path)

    random.shuffle(all_aggregated_data)
    final_subset = all_aggregated_data[:n_samples]
    save_to_csv(final_subset, f"{save_path}/box_all_entities_instruct.csv")
    

def create_inverse_country_dataset(tokenizer, n_samples, n_entities=10, random_seed=11, save_path="data/instruct_country_inverse", dataset_type="country"):
    random.seed(random_seed)
    valid_items = filter_single_tokens(tokenizer, COUNTRIES, prepend_space=True) 
    valid_entities = filter_single_tokens(tokenizer, NAMES)
    
    all_aggregated_data = []

    for n in range(2, n_entities + 1):
        data = []
        attempts = 0
        while len(data) < n_samples and attempts < n_samples * 5:
            attempts += 1
            sampled_items = random.sample(valid_items, n)
            sampled_entities = random.sample(valid_entities, n)
            
            sample = get_inverse_country_sample(
                tokenizer, n, sampled_entities, sampled_items, 
                valid_items, valid_entities, dataset_type=dataset_type
            )
            
            if not any(d['clean'] == sample['clean'] for d in data):
                data.append(sample)
                
        all_aggregated_data.extend(data)

        if dataset_type == "country-filler-related":
            fname = f"country_{n}_entities_filler.csv"
        elif dataset_type == "country-filler-unrelated":
            fname = f"country_{n}_entities_filler_unrelated.csv"
        else:
            fname = f"country_{n}_entities.csv"
            
        save_to_csv(data, f"{save_path}/{fname}")

    random.shuffle(all_aggregated_data)
    final_subset = all_aggregated_data[:n_samples]
    save_to_csv(final_subset, f"{save_path}/country_all_entities.csv")


def create_inverse_positional_country_datasets(tokenizer, n_samples, random_seed=11, save_path="data/instruct_country_positional"):
    random.seed(random_seed)
    n_entities = 9 

    valid_items = filter_single_tokens(tokenizer, COUNTRIES, prepend_space=True) 
    valid_entities = filter_single_tokens(tokenizer, NAMES)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for pos_index in range(n_entities):
        data = []
        attempts = 0
        
        while len(data) < n_samples and attempts < n_samples * 5:
            attempts += 1
            
            sampled_items = random.sample(valid_items, n_entities)
            sampled_entities = random.sample(valid_entities, n_entities)
            
            sample = get_inverse_fixed_position_sample(
                tokenizer, 
                n_entities, 
                sampled_entities, 
                sampled_items, 
                valid_entities,
                target_pos_index=pos_index 
            )
            
            if not any(d['clean'] == sample['clean'] for d in data):
                data.append(sample)
        
        filename = f"{save_path}/country_pos_{pos_index}.csv"
        save_to_csv(data, filename)

