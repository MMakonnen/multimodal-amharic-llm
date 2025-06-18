from transformers import AutoTokenizer

# --- Configuration ---
# Path where the custom tokenizer is saved
EXTENDED_TOKENIZER_SAVE_PATH = "./llama3_tokenizer_amharic_extended_3sources"
# The identifier for the original Llama 3 model on Hugging Face
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"  
# Amharic text for tokenization comparison
AMHARIC_TEXT = "ሰላም፣ እንዴት ነህ?" # Translation: "Hello how are you?"

# --- 1. Compare with the Original Llama 3 Tokenizer ---
print("--- Original Llama 3 Tokenizer ---")
try:
    # Load the original Llama 3 tokenizer
    original_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    print(f"Successfully loaded original Llama 3 tokenizer. Vocab size: {len(original_tokenizer)}")

    # Tokenize and encode the Amharic text
    original_tokens = original_tokenizer.tokenize(AMHARIC_TEXT)
    print(f"Tokenizing Amharic text '{AMHARIC_TEXT}': {original_tokens}")
    
    # show the token IDs for the Amharic text
    original_token_ids = original_tokenizer.encode(AMHARIC_TEXT, add_special_tokens=True)
    print(f"Token IDs for Amharic text '{AMHARIC_TEXT}': {original_token_ids}")
                                                   

except Exception as e:
    print(f"Error loading or using the original Llama 3 tokenizer: {e}")
    print("Please ensure you have authenticated with Hugging Face CLI: `huggingface-cli login`")

print("\n" + "="*50 + "\n")

# --- 2. Load and test the Extended Tokenizer ---
print("--- Extended Llama 3 Tokenizer (Amharic) ---")
try:
    # Load your custom, extended tokenizer from the local path
    loaded_extended_tokenizer = AutoTokenizer.from_pretrained(EXTENDED_TOKENIZER_SAVE_PATH)
    print(f"Successfully loaded extended tokenizer. Vocab size: {len(loaded_extended_tokenizer)}")

    # Tokenize and encode the same Amharic text
    extended_tokens = loaded_extended_tokenizer.tokenize(AMHARIC_TEXT)
    print(f"Tokenizing Amharic text '{AMHARIC_TEXT}': {extended_tokens}")

    # show the token IDs for the Amharic text
    extended_token_ids = loaded_extended_tokenizer.encode(AMHARIC_TEXT, add_special_tokens=True)
    print(f"Token IDs for Amharic text '{AMHARIC_TEXT}': {extended_token_ids}")

except Exception as e:
    print(f"Error loading or using the extended tokenizer: {e}")
    print(f"Please ensure the path '{EXTENDED_TOKENIZER_SAVE_PATH}' contains the tokenizer files.")

# --- 3. Load and test another tokenizer (optional) ---
print("\n" + "="*50 + "\n")
print("--- Testing Another Tokenizer (if available) ---")
try:
    # Load another tokenizer, e.g., a different model or a custom one
    another_tokenizer = AutoTokenizer.from_pretrained("rasyosef/Llama-3.2-1B-Amharic")
    print(f"Successfully loaded another tokenizer. Vocab size: {len(another_tokenizer)}")

    # Tokenize the same Amharic text
    another_tokens = another_tokenizer.tokenize(AMHARIC_TEXT)
    print(f"Tokenizing Amharic text '{AMHARIC_TEXT}' with another tokenizer: {another_tokens}")

    # show the token IDs for the Amharic text
    another_token_ids = another_tokenizer.encode(AMHARIC_TEXT, add_special_tokens=True)
    print(f"Token IDs for Amharic text '{AMHARIC_TEXT}' with another tokenizer: {another_token_ids}")
                                                 
except Exception as e:
    print(f"Error loading or using another tokenizer: {e}")
    print("This step is optional and can be skipped if not needed.")