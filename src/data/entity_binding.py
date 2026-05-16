import pdb
import torch as t
import random

from src.utils.general_utils import is_correct
from src.utils import general_utils as gu

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

COLORS = ["red", "yellow", "green", "orange", "blue", "black", "white", "purple", "violet", "gold", "silver", "grey", "brown", "pink", "teal"]

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

def filter_single_tokens(tokenizer, word_list, prepend_space=False, contains_question=False):
    """Filters a list to keep only words that tokenize to a single token."""
    valid_words = []
    for word in word_list:
        text = " " + word if prepend_space else word
        text = text + "?" if contains_question else text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if contains_question:
            if len(tokens) == 2:
                valid_words.append(word)
        else:
            if len(tokens) == 1:
                valid_words.append(word)
        
    return valid_words

def save_to_csv(data, path):
    """
    Saves list of dicts to CSV, STRICTLY enforcing the required columns.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    df = pd.DataFrame(data)
    
    # STRICTLY FORCE COLUMNS
    required_cols = ['clean', 'corrupted', 'correct_idx', 'incorrect_idx', 'label', 'corrupted_labels']
    df = df[required_cols]
    
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def box_comma_prompt(sampled_items, sampled_letters, n_entity):
    prompt = ""
    i = 0
    for item, letter in zip(sampled_items, sampled_letters):
        i += 1
        if i == 1:
            prompt += f"The {item} is in box {letter}, "
        elif i == n_entity:
            prompt += f"the {item} is in box {letter}. "
        else:
            prompt += f"the {item} is in box {letter}, "

    return prompt

def box_period_prompt(sampled_items, sampled_letters, n_entity):
    prompt = ""
    i = 0
    for item, letter in zip(sampled_items, sampled_letters):
        i += 1
        if i == 1:
            prompt += f"The {item} is in box {letter}. "
        elif i == n_entity:
            prompt += f"The {item} is in box {letter}. "
        else:
            prompt += f"The {item} is in box {letter}. "

    return prompt

def box_colon_prompt(sampled_items, sampled_letters, n_entity):
    prompt = ""
    i = 0
    for item, letter in zip(sampled_items, sampled_letters):
        i += 1
        if i == 1:
            prompt += f"The {item} is in box {letter}; "
        elif i == n_entity:
            prompt += f"the {item} is in box {letter}. "
        else:
            prompt += f"the {item} is in box {letter}; "

    return prompt

def country_comma_prompt(sampled_countries, sampled_person, n_entity):
    prompt = ""
    i = 0
    for country, person in zip(sampled_countries, sampled_person):
        i += 1
        if i == 1:
            prompt += f"{person} lives in {country}, "
        elif i == n_entity:
            prompt += f"{person} lives in {country}. "
        else:
            prompt += f"{person} lives in {country}, "

    return prompt

def country_period_prompt(sampled_countries, sampled_person, n_entity):
    prompt = ""
    i = 0
    for item, letter in zip(sampled_countries, sampled_person):
        i += 1
        if i == 1:
            prompt += f"{person} lives in {country}. "
        elif i == n_entity:
            prompt += f"{person} lives in {country}. "
        else:
            prompt += f"{person} lives in {country}. "

    return prompt

def country_colon_prompt(sampled_countries, sampled_person, n_entity):
    prompt = ""
    i = 0
    for item, letter in zip(sampled_countries, sampled_person):
        i += 1
        if i == 1:
            prompt += f"{person} lives in {country}; "
        elif i == n_entity:
            prompt += f"{person} lives in {country}. "
        else:
            prompt += f"{person} lives in {country}; "

    return prompt

def color_box_comma_prompt(sampled_items, sampled_colors, n_entity):
    prompt = ""
    i = 0
    for item, color in zip(sampled_items, sampled_colors):
        i += 1
        if i == 1:
            prompt += f"The {item} is in {color} box, "
        elif i == n_entity:
            prompt += f"the {item} is in {color} box. "
        else:
            prompt += f"the {item} is in {color} box, "

    return prompt

def create_box_data(
    model,
    tokenizer, 
    n_samples: int, 
    data_type: str,
    n_entity: int = 2,
    only_correct: bool = True
    ):
    valid_items = filter_single_tokens(tokenizer, HOUSEHOLD_ITEMS)
    valid_letters = filter_single_tokens(tokenizer, LETTERS)
    valid_colors = filter_single_tokens(tokenizer, COLORS)

    data = []
    attempts = 0
    while len(data) < n_samples and attempts < n_samples * 5:
        attempts += 1
        sampled_items = random.sample(valid_items, n_entity)
        sampled_letters = random.sample(valid_letters, n_entity)
        sampled_colors = random.sample(valid_colors, n_entity)
        if data_type == "comma":
            prompt = box_comma_prompt(sampled_items, sampled_letters, n_entity)
        elif data_type == "period":
            prompt = box_period_prompt(sampled_items, sampled_letters, n_entity)
        elif data_type == "colon":
            prompt = box_colon_prompt(sampled_items, sampled_letters, n_entity)
        elif data_type == "instruct":
            prompt = box_comma_prompt(sampled_items, sampled_letters, n_entity)
        elif "color" in data_type:
            prompt = color_box_comma_prompt(sampled_items, sampled_colors, n_entity)
        
        if "position" in data_type:
            if "comma" in data_type:
                prompt = box_comma_prompt(sampled_items, sampled_letters, n_entity)
            selected_index = int(data_type.split("-")[1])
        else:
            selected_index = random.randint(0, n_entity - 1)
        # counterfactual_index = random.randint(0, n_entity - 1)
        # while counterfactual_index == selected_index:
        #     counterfactual_index = random.randint(0, n_entity - 1)
        
        counterfactual_letter = random.choice([letter for letter in valid_letters if letter not in sampled_letters])
        counterfactual_color = random.choice([color for color in valid_colors if color not in sampled_colors])
        if data_type == "instruct":
            clean_prompt = "Question: "+ prompt + f"What does box {sampled_letters[selected_index]} contain? Answer:"
            counterfactual_prompt = "Question: " + prompt + f"What does box {counterfactual_letter} contain? Answer:"
        elif "color" in data_type:
            clean_prompt = prompt + f"{sampled_colors[selected_index]} box contains the".capitalize()
            counterfactual_prompt = prompt + f"{counterfactual_color} box contains the".capitalize()
        else:
            clean_prompt = prompt + f"Box {sampled_letters[selected_index]} contains the"
            counterfactual_prompt = prompt + f"Box {counterfactual_letter} contains the"
        
        label = " " + sampled_items[selected_index]
        corrupted_labels = [" " + sampled_items[idx] for idx in range(n_entity) if idx != selected_index]
        correct_idx = tokenizer.encode(label, add_special_tokens=False)
        incorrect_idx = [tokenizer.encode(corrupted_label, add_special_tokens=False)[0] for corrupted_label in corrupted_labels]
        assert len(correct_idx) == 1, f"Tokenization error: {correct_idx}, {incorrect_idx}"
        
        tmp = {
            "clean": clean_prompt, 
            "corrupted": counterfactual_prompt, 
            "correct_idx": correct_idx[0], 
            "incorrect_idx": incorrect_idx, 
            "label": sampled_items[selected_index], 
            "corrupted_labels": [item for idx, item in enumerate(sampled_items) if idx != selected_index]
        }
        if only_correct:
            with t.no_grad():
                logits = model(tmp["clean"], return_type="logits")
            if is_correct(logits, tmp["correct_idx"], tmp["incorrect_idx"]) is False:
                print("Model got it wrong, skipping this sequence.")
                continue  # Skip this sequence if the model got it wrong
        data.append(tmp)
    return data


def create_country_data(
    model,
    tokenizer, 
    n_samples: int, 
    data_type: str,
    n_entity: int = 2,
    only_correct: bool = True
    ):
    valid_countries = filter_single_tokens(tokenizer, COUNTRIES, prepend_space=True, contains_question=True)
    valid_names = filter_single_tokens(tokenizer, NAMES, prepend_space=True, contains_question=True)

    data = []
    attempts = 0
    while len(data) < n_samples and attempts < n_samples * 5:
        attempts += 1
        sampled_countries = random.sample(valid_countries, n_entity)
        sampled_names = random.sample(valid_names, n_entity)
        if data_type == "comma":
            prompt = country_comma_prompt(sampled_countries, sampled_names, n_entity)
        elif data_type == "period":
            prompt = country_period_prompt(sampled_countries, sampled_names, n_entity)
        elif data_type == "colon":
            prompt = country_colon_prompt(sampled_countries, sampled_names, n_entity)
        selected_index = random.randint(0, n_entity - 1)
        # counterfactual_index = random.randint(0, n_entity - 1)
        # while counterfactual_index == selected_index:
        #     counterfactual_index = random.randint(0, n_entity - 1)
        
        clean_prompt = "Question: " + prompt + f"Who lives in {sampled_countries[selected_index]}? Answer:"
        counterfactual_country = random.choice([country for country in valid_countries if country not in sampled_countries])
        counterfactual_prompt = "Question: " + prompt + f"Who lives in {counterfactual_country}? Answer:" 
        label = " " + sampled_names[selected_index]
        corrupted_labels = [" " + sampled_names[idx] for idx in range(n_entity) if idx != selected_index]
        correct_idx = tokenizer.encode(label, add_special_tokens=False)
        incorrect_idx = [tokenizer.encode(corrupted_label, add_special_tokens=False)[0] for corrupted_label in corrupted_labels]
        assert len(correct_idx) == 1, f"Tokenization error: {correct_idx}, {incorrect_idx}"
        
        tmp = {
            "clean": clean_prompt, 
            "corrupted": counterfactual_prompt, 
            "correct_idx": correct_idx[0], 
            "incorrect_idx": incorrect_idx, 
            "label": sampled_names[selected_index], 
            "corrupted_labels": [item for idx, item in enumerate(sampled_names) if idx != selected_index]
        }
        if only_correct:
            with t.no_grad():
                logits = model(tmp["clean"], return_type="logits")
            if is_correct(logits, tmp["correct_idx"], tmp["incorrect_idx"]) is False:
                print("Model got it wrong, skipping this sequence.")
                continue  # Skip this sequence if the model got it wrong
        data.append(tmp)
    return data

def create_data(model, tokenizer, dataset_type: str, n_data: int = 1000, only_correct: bool = True, random_seed: int = 32):
    # Pin global random state so random.sample/randint/choice inside
    # create_box_data and create_country_data are deterministic.
    random.seed(random_seed)
    if "comma" in dataset_type:
        if dataset_type == "comma":
            data = create_box_data(model, tokenizer, n_samples=n_data, data_type="comma", n_entity=2, only_correct=only_correct)
        elif "position" in dataset_type:
            data = create_box_data(model, tokenizer, n_samples=n_data, data_type=dataset_type, n_entity=8, only_correct=only_correct)
        elif "color" in dataset_type:
            n_entity = int(dataset_type.split("-")[0])
            data = create_box_data(model, tokenizer, n_samples=n_data, data_type=dataset_type, n_entity=n_entity, only_correct=only_correct)
        else:
            n_entity = int(dataset_type.split("-")[0])
            data = create_box_data(model, tokenizer, n_samples=n_data, data_type="comma", n_entity=n_entity, only_correct=only_correct)
    
        
    
    elif "period" in dataset_type:
        if dataset_type == "period":
            data = create_box_data(model, tokenizer, n_samples=n_data, data_type="period", n_entity=2, only_correct=only_correct)
        else:
            n_entity = int(dataset_type.split("-")[0])
            data = create_box_data(model, tokenizer, n_samples=n_data, data_type="period", n_entity=n_entity, only_correct=only_correct)
    
    elif "instruct" in dataset_type:
        if dataset_type == "instruct":
            data = create_box_data(model, tokenizer, n_samples=n_data, data_type="instruct", n_entity=2, only_correct=only_correct)
        else:
            n_entity = int(dataset_type.split("-")[0])
            data = create_box_data(model, tokenizer, n_samples=n_data, data_type="instruct", n_entity=n_entity, only_correct=only_correct)

    elif dataset_type == "colon":
        data = create_box_data(model, tokenizer, n_samples=n_data, data_type="colon", n_entity=2, only_correct=only_correct) 

    elif "country" in dataset_type:
        if dataset_type == "country":
            data = create_country_data(model, tokenizer, n_samples=n_data, data_type="comma", n_entity=2, only_correct=only_correct)
        else:
            n_entity = int(dataset_type.split("-")[0])
            data = create_country_data(model, tokenizer, n_samples=n_data, data_type="comma", n_entity=n_entity, only_correct=only_correct)
    else:
        raise ValueError(f"Unknown type: {dataset_type}")

    return data

