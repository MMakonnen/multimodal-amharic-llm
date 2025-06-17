from transformers import AutoTokenizer

# --- Configuration ---
# Path where the custom tokenizer is saved
EXTENDED_TOKENIZER_SAVE_PATH = "./llama3_tokenizer_amharic_extended"
# The identifier for the original Llama 3 model on Hugging Face
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"  
# Amharic text for tokenization comparison
AMHARIC_TEXT = "ሰላም፣ እንዴት ነህ?" # Translation: "This is an Amharic test sentence."

# --- 1. Compare with the Original Llama 3 Tokenizer ---
print("--- Original Llama 3 Tokenizer ---")
try:
    # Load the original Llama 3 tokenizer
    original_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    print(f"Successfully loaded original Llama 3 tokenizer. Vocab size: {len(original_tokenizer)}")

    # Tokenize and encode the Amharic text
    original_tokens = original_tokenizer.tokenize(AMHARIC_TEXT)
    print(f"Tokenizing Amharic text '{AMHARIC_TEXT}': {original_tokens}")
    
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

except Exception as e:
    print(f"Error loading or using the extended tokenizer: {e}")
    print(f"Please ensure the path '{EXTENDED_TOKENIZER_SAVE_PATH}' contains the tokenizer files.")