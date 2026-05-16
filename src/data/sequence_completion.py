import random
import torch as t

from src.utils.general_utils import is_correct

NAMES = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer",
    "Joshua", "Amanda", "Daniel", "David", "James", "Robert", "John",
    "Joseph", "Andrew", "Ryan", "Brandon", "Jason", "Justin", "Sarah",
    "William", "Jonathan", "Stephanie", "Brian", "Nicole", "Nicholas",
    "Anthony", "Heather", "Eric", "Elizabeth", "Adam", "Megan", "Melissa",
    "Kevin", "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel",
    "Laura", "Lauren", "Amber", "Brittany", "Danielle", "Richard",
    "Kimberly", "Jeffrey", "Amy", "Crystal", "Michelle", "Tiffany",
    "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles", "Rebecca",
    "Jacob", "Stephen", "Patrick", "Sean", "Erin", "Jamie", "Kelly",
    "Samantha", "Nathan", "Sara", "Dustin", "Paul", "Angela", "Tyler",
    "Scott", "Katherine", "Andrea", "Gregory", "Erica", "Mary", "Travis",
    "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen", "Jose", "Alexander",
    "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney",
    "Christine", "Alicia", "Cody", "Allison", "Bradley", "Samuel",
]


def single_token(model):
    """Filter names to those that tokenize as a single token."""
    single_names = [
        name for name in NAMES
        if len(model.tokenizer.tokenize(name)) == 1
        and len(model.tokenizer.tokenize(" " + name)) == 1
    ]
    return single_names


def sample_tokens(model, n_tokens: int, person_name: bool = True):
    """
    Sample n unique tokens, returning them as strings.
    
    Args:
        model: The model (for tokenizer and vocab size)
        n_tokens: Number of tokens to sample
        person_name: If True, sample from single-token names; else sample from vocab
    
    Returns:
        List of string tokens
    """
    if person_name:
        single_names = single_token(model)
        if len(single_names) < n_tokens:
            raise ValueError(f"Not enough single-token names. Need {n_tokens}, have {len(single_names)}")
        return random.sample(single_names, n_tokens)
    else:
        vocab_size = model.cfg.d_vocab
        token_ids = random.sample(range(vocab_size), n_tokens)
        # Convert token IDs to strings for consistent handling
        return [model.tokenizer.decode([tid]).strip() for tid in token_ids]


def create_symbolic_data(
    model, 
    tokenizer,
    seq_len: int = 6,
    n_gram: int = 2,
    n_data: int = 1000,
    repeat_n: int = 4,
    only_correct: bool = True,
    person_name: bool = True,
    random_seed: int = None
):
    """
    Generates sequences to test n-gram induction heads with repetitions.
    
    Pattern design (seq_len=6):
    - 2-gram: [A B C D E F] - model looks back 1 token (E → F)
    - 3-gram: [A B C X E F] vs [A B C Y E G] - model looks back 2 tokens (X E → F vs Y E → G)
    - 4-gram: [A B X C E F] vs [A B Y C E G] - model looks back 3 tokens (X C E → F vs Y C E → G)
    
    Args:
        model: The model (for vocab size and BOS token)
        tokenizer: The tokenizer
        seq_len: Length of each pattern (default 6)
        n_gram: The n in n-gram (2, 3, or 4)
        n_data: Number of data sequences to generate
        repeat_n: Number of times to repeat patterns (default 4)
        only_correct: If True, only keep sequences the model predicts correctly
        person_name: If True, use human names; else random vocab tokens
        random_seed: Random seed for reproducibility
    
    Returns:
        List of dictionaries containing test sequences
    """
    if random_seed is not None:
        random.seed(random_seed)
        t.manual_seed(random_seed)
    
    all_sequences = []

    while len(all_sequences) < n_data:
        tmp = {}
        
        if n_gram == 2:
            # Pattern: A B C D E F
            # Test: E → F (1 token context)
            tokens = sample_tokens(model, seq_len, person_name)
            A, B, C, D, E, F = tokens[:6]
            
            pattern = [A, B, C, D, E, F]
            
            # For counterfactual, we break the E → F association
            # Use different tokens for the last two positions
            cf_tokens = sample_tokens(model, 2, person_name)
            while cf_tokens[0] in tokens or cf_tokens[1] in tokens:
                cf_tokens = sample_tokens(model, 2, person_name)
            
            pattern_counterfactual = [A, B, C, D, cf_tokens[0], cf_tokens[1]]
            
            sequence = pattern * repeat_n
            counterfactual_sequence = pattern_counterfactual * repeat_n
            
        elif n_gram == 3:
            # Pattern X: A B C X E F
            # Pattern Y: A B C Y E G
            # Test: X E → F (2 token context)
            # Model must look at (X, E) to predict F vs G
            tokens = sample_tokens(model, 10, person_name)
            A, B, C, X, Y, E, F, G, C1, C2 = tokens[:10]
            
            pattern_X = [A, B, C, X, E, F]
            pattern_Y = [A, B, C, Y, E, G]
            
            # Counterfactual: replace X with C1 to break the X E → F association
            pattern_counterfactual = [A, B, C, C1, E, C2]
            
            # Build sequence: 2 of each pattern (4 total), then test
            patterns = [pattern_X, pattern_Y] * 2
            random.shuffle(patterns)
            
            sequence = []
            for p in patterns:
                sequence.extend(p)
            # Add test sequence (with final token F)
            sequence.extend([A, B, C, X, E, F])
            
            # Counterfactual sequence
            counterfactual_sequence = []
            for p in patterns:
                if p == pattern_X:
                    counterfactual_sequence.extend(pattern_counterfactual)
                else:
                    counterfactual_sequence.extend(p)
            counterfactual_sequence.extend(pattern_counterfactual)
                
        elif n_gram == 4:
            # Pattern X: A B X C E F
            # Pattern Y: A B Y C E G
            # Test: X C E → F (3 token context)
            # Model must look at (X, C, E) to predict F vs G
            tokens = sample_tokens(model, 10, person_name)
            A, B, X, Y, C, E, F, G, C1, C2 = tokens[:10]
            
            pattern_X = [A, B, X, C, E, F]
            pattern_Y = [A, B, Y, C, E, G]
            
            # Counterfactual: replace X with C1 to break the X C E → F association
            pattern_counterfactual = [A, B, C1, C, E, C2]
            
            # Build sequence: 2 of each pattern (4 total), then test
            patterns = [pattern_X, pattern_Y] * 2
            random.shuffle(patterns)
            
            sequence = []
            for p in patterns:
                sequence.extend(p)
            # Add test sequence
            sequence.extend([A, B, X, C, E, F])
            
            # Counterfactual sequence
            counterfactual_sequence = []
            for p in patterns:
                if p == pattern_X:
                    counterfactual_sequence.extend(pattern_counterfactual)
                else:
                    counterfactual_sequence.extend(p)
            counterfactual_sequence.extend(pattern_counterfactual)
        
        else:
            raise ValueError(f"Only n=2, 3, 4 are supported, got n={n_gram}")
        
        # Build the data dictionary
        tmp["clean"] = " ".join(sequence[:-1])
        tmp["corrupted"] = " ".join(counterfactual_sequence[:-1])
        
        tmp["correct_idx"] = model.tokenizer.encode(" " + sequence[-1], add_special_tokens=False)[0]
        tmp["incorrect_idx"] = [model.tokenizer.encode(" " + tok, add_special_tokens=False)[0] for tok in tokens if tok != sequence[-1]]
        tmp["label"] = " " + sequence[-1]
        tmp["corrupted_labels"] = [" " + tok for tok in tokens]

        clean_len = len(model.tokenizer.encode(tmp["clean"]))
        corrupted_len = len(model.tokenizer.encode(tmp["corrupted"]))
        assert clean_len == corrupted_len, (
            f"Token length mismatch: clean={clean_len}, corrupted={corrupted_len}. "
            f"Clean: {tmp['clean']!r}, Corrupted: {tmp['corrupted']!r}"
        )

        if only_correct:
            with t.no_grad():
                logits = model(tmp["clean"], return_type="logits")

            if not is_correct(logits, tmp["correct_idx"], tmp["incorrect_idx"]):
                print("Model got it wrong, skipping this sequence.")
                continue

        all_sequences.append(tmp)
    
    return all_sequences


def create_data(
    model, 
    tokenizer, 
    prompt_type: str, 
    n_data: int = 1000, 
    only_correct: bool = False, 
    random_seed: int = 32
):
    """
    Main entry point for creating test data.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompt_type: Format "N-gram-symbolic" where N is 2, 3, or 4
        n_data: Number of sequences to generate
        only_correct: If True, only keep sequences model predicts correctly
        random_seed: Random seed for reproducibility
    
    Returns:
        List of data dictionaries
    """
    n_gram = int(prompt_type.split("-")[0])
    
    if "symbolic" in prompt_type:
        return create_symbolic_data(
            model, 
            tokenizer, 
            n_gram=n_gram, 
            n_data=n_data, 
            only_correct=only_correct,
            random_seed=random_seed
        )
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")


# ============================================================================
# ASSERT TESTS
# ============================================================================

def test_2gram_pattern():
    """
    2-gram test:
    Pattern: A B C D E F (repeated 4 times)
    Input:  "A B C D E F A B C D E F A B C D E F A B C D E"
    Output: "F"
    
    Model looks back 1 token: E → F
    """
    A, B, C, D, E, F = "A", "B", "C", "D", "E", "F"
    pattern = [A, B, C, D, E, F]
    
    sequence = pattern * 4
    
    input_seq = " ".join(sequence[:-1])
    expected_output = F
    
    expected_input = "A B C D E F A B C D E F A B C D E F A B C D E"
    
    assert input_seq == expected_input, f"Input mismatch:\n  Got: {input_seq}\n  Expected: {expected_input}"
    assert sequence[-1] == expected_output, f"Output mismatch: Got {sequence[-1]}, Expected {expected_output}"
    
    print("✓ 2-gram pattern test passed")
    return input_seq, expected_output


def test_3gram_pattern():
    """
    3-gram test:
    Pattern X: A B C X E F
    Pattern Y: A B C Y E G
    
    Sequence (4 reps of patterns + test): X, Y, X, Y, then X E → ?
    Input:  "A B C X E F A B C Y E G A B C X E F A B C Y E G A B C X E"
    Output: "F"
    
    Model looks back 2 tokens: X E → F (vs Y E → G)
    """
    A, B, C, X, Y, E, F, G = "A", "B", "C", "X", "Y", "E", "F", "G"
    
    pattern_X = [A, B, C, X, E, F]
    pattern_Y = [A, B, C, Y, E, G]
    
    # Fixed order for test: X, Y, X, Y, then test with X
    sequence = pattern_X + pattern_Y + pattern_X + pattern_Y + [A, B, C, X, E, F]
    
    input_seq = " ".join(sequence[:-1])
    expected_output = F
    
    expected_input = "A B C X E F A B C Y E G A B C X E F A B C Y E G A B C X E"
    
    assert input_seq == expected_input, f"Input mismatch:\n  Got: {input_seq}\n  Expected: {expected_input}"
    assert sequence[-1] == expected_output, f"Output mismatch: Got {sequence[-1]}, Expected {expected_output}"
    
    print("✓ 3-gram pattern test passed")
    return input_seq, expected_output


def test_4gram_pattern():
    """
    4-gram test:
    Pattern X: A B X C E F
    Pattern Y: A B Y C E G
    
    Sequence (4 reps of patterns + test): X, Y, X, Y, then X C E → ?
    Input:  "A B X C E F A B Y C E G A B X C E F A B Y C E G A B X C E"
    Output: "F"
    
    Model looks back 3 tokens: X C E → F (vs Y C E → G)
    """
    A, B, X, Y, C, E, F, G = "A", "B", "X", "Y", "C", "E", "F", "G"
    
    pattern_X = [A, B, X, C, E, F]
    pattern_Y = [A, B, Y, C, E, G]
    
    # Fixed order for test: X, Y, X, Y, then test with X
    sequence = pattern_X + pattern_Y + pattern_X + pattern_Y + [A, B, X, C, E, F]
    
    input_seq = " ".join(sequence[:-1])
    expected_output = F
    
    expected_input = "A B X C E F A B Y C E G A B X C E F A B Y C E G A B X C E"
    
    assert input_seq == expected_input, f"Input mismatch:\n  Got: {input_seq}\n  Expected: {expected_input}"
    assert sequence[-1] == expected_output, f"Output mismatch: Got {sequence[-1]}, Expected {expected_output}"
    
    print("✓ 4-gram pattern test passed")
    return input_seq, expected_output


def test_context_requirements():
    """
    Test that the patterns actually require the correct n-gram context.
    
    For 3-gram: The token before E (X vs Y) determines the output (F vs G)
    For 4-gram: The token 2 positions before E (X vs Y) determines the output (F vs G)
    """
    # 3-gram: ambiguity at 1-token context, resolved at 2-token context
    pattern_3gram_X = ["A", "B", "C", "X", "E", "F"]
    pattern_3gram_Y = ["A", "B", "C", "Y", "E", "G"]
    
    # Both end with E, so 1-token context (E) is ambiguous
    assert pattern_3gram_X[-2] == pattern_3gram_Y[-2] == "E", "3-gram: Both patterns should share E"
    # But (X, E) vs (Y, E) distinguishes them
    assert pattern_3gram_X[-3] != pattern_3gram_Y[-3], "3-gram: X and Y should differ"
    
    # 4-gram: ambiguity at 2-token context, resolved at 3-token context
    pattern_4gram_X = ["A", "B", "X", "C", "E", "F"]
    pattern_4gram_Y = ["A", "B", "Y", "C", "E", "G"]
    
    # Both share (C, E) suffix, so 2-token context is ambiguous
    assert pattern_4gram_X[-2] == pattern_4gram_Y[-2] == "E", "4-gram: Both patterns should share E"
    assert pattern_4gram_X[-3] == pattern_4gram_Y[-3] == "C", "4-gram: Both patterns should share C"
    # But (X, C, E) vs (Y, C, E) distinguishes them
    assert pattern_4gram_X[-4] != pattern_4gram_Y[-4], "4-gram: X and Y should differ"
    
    print("✓ Context requirement test passed")


def run_all_tests():
    """Run all pattern tests."""
    print("Running n-gram pattern tests...\n")
    
    test_2gram_pattern()
    test_3gram_pattern()
    test_4gram_pattern()
    test_context_requirements()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    run_all_tests()