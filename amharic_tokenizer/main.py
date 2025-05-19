import os
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import SentencePieceBPETokenizer # For training the new Amharic vocab component

# Step 1: Define Parameters
LLAMA3_MODEL_ID = "meta-llama/Llama-3.2-1B"  # REPLACE with your Llama 3 model
AMHARIC_DATASET_ID = "iocuydi/amharic-redpajama-synthetic"
EXTENDED_TOKENIZER_SAVE_PATH = "./llama3_tokenizer_amharic_extended"
# How many Amharic-specific tokens to try and learn.
# This is for the *new* SentencePiece model trained on Amharic data.
# The final Llama3 tokenizer will be its original vocab + these (approximately).
NEW_AMHARIC_VOCAB_SIZE = 10000 # Adjust based on dataset size and needs
# How much of the dataset to use for training the Amharic SentencePiece model.
# Using the full dataset can be slow; adjust based on your resources.
# For very large datasets, you might use a representative subset.
DATASET_SAMPLE_FOR_TOKENIZER_TRAINING = 86350170//4 # Number of samples

# Step 2: Load the Amharic Dataset
print(f"Loading Amharic dataset: {AMHARIC_DATASET_ID}")
try:
    # Attempt to load a specific number of samples for tokenizer training efficiency
    dataset = load_dataset(AMHARIC_DATASET_ID, split=f"train[:{DATASET_SAMPLE_FOR_TOKENIZER_TRAINING}]", streaming=False, cache_dir="./cache")
except ValueError: # Handle cases where the dataset might be smaller than the sample size
    print(f"Could not load {DATASET_SAMPLE_FOR_TOKENIZER_TRAINING} samples. Loading entire 'train' split.")
    dataset = load_dataset(AMHARIC_DATASET_ID, split="train", streaming=False)
except Exception as e:
    print(f"Error loading dataset {AMHARIC_DATASET_ID}: {e}")
    exit()

print(f"Loaded {len(dataset)} samples for tokenizer training.")

# Create a text iterator from the dataset
def get_text_iterator(data):
    print("Preparing text iterator...")
    count = 0
    for item in data:
        if 'text' in item and isinstance(item['text'], str):
            yield item['text']
            count +=1
        elif 'prompt' in item and isinstance(item['prompt'], str): # Common alternative field name
            yield item['prompt']
            count +=1
        else: # Fallback: try to find any string field
            for key in item:
                if isinstance(item[key], str):
                    yield item[key]
                    count +=1
                    break
    print(f"Text iterator will yield {count} text samples.")

# Step 3: Train a New SentencePiece Tokenizer on Amharic Data
# This helps identify Amharic-specific subwords.
print(f"\nTraining a new SentencePiece BPE tokenizer for Amharic with vocab_size={NEW_AMHARIC_VOCAB_SIZE}...")
amharic_sp_tokenizer = SentencePieceBPETokenizer()

# Train the SentencePiece model
amharic_sp_tokenizer.train_from_iterator(
    get_text_iterator(dataset),
    vocab_size=NEW_AMHARIC_VOCAB_SIZE,
    min_frequency=2, # A common setting
    show_progress=True,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"] # Standard special tokens
)

newly_learned_amharic_tokens = list(amharic_sp_tokenizer.get_vocab().keys())
print(f"Trained Amharic SentencePiece tokenizer. Learned {len(newly_learned_amharic_tokens)} tokens.")

# Step 4: Load the Base Llama 3 Tokenizer
print(f"\nLoading base Llama 3 tokenizer from: {LLAMA3_MODEL_ID}")
try:
    llama3_base_tokenizer = AutoTokenizer.from_pretrained(LLAMA3_MODEL_ID)
except Exception as e:
    print(f"Error loading base Llama 3 tokenizer: {e}")
    print("Ensure you have access to the model (e.g., run huggingface-cli login).")
    exit()

original_vocab_size = llama3_base_tokenizer.vocab_size
print(f"Base Llama 3 tokenizer loaded. Original vocabulary size: {original_vocab_size}")

# Step 5: Add the New Amharic Tokens to the Llama 3 Tokenizer
print(f"\nAdding {len(newly_learned_amharic_tokens)} learned Amharic tokens to the Llama 3 tokenizer...")
# The add_tokens method returns the number of tokens actually added (it handles duplicates)
num_tokens_added = llama3_base_tokenizer.add_tokens(newly_learned_amharic_tokens)
print(f"Successfully added {num_tokens_added} new tokens.")

extended_vocab_size = llama3_base_tokenizer.vocab_size
print(f"Llama 3 tokenizer vocabulary size after extension: {extended_vocab_size}")

# Step 6: Save the Extended Tokenizer
os.makedirs(EXTENDED_TOKENIZER_SAVE_PATH, exist_ok=True)
print(f"\nSaving extended tokenizer to: {EXTENDED_TOKENIZER_SAVE_PATH}")
llama3_base_tokenizer.save_pretrained(EXTENDED_TOKENIZER_SAVE_PATH)
print("Extended tokenizer saved successfully.")

# --- CRITICAL NEXT STEPS (Conceptual - Not run by this script) ---
print("\n--- Tokenizer Extension Process Finished ---")
print(f"Original Llama 3 tokenizer vocab size: {original_vocab_size}")
print(f"Extended tokenizer vocab size: {extended_vocab_size} ({num_tokens_added} new tokens added)")
print(f"The extended tokenizer is saved at: {EXTENDED_TOKENIZER_SAVE_PATH}")

print("\nVERY IMPORTANT NEXT STEPS:")
print("1. Load the Llama 3 *model* (e.g., AutoModelForCausalLM.from_pretrained(LLAMA3_MODEL_ID)).")
print("2. Resize the model's token embeddings to match the new tokenizer size:")
print(f"   model.resize_token_embeddings({extended_vocab_size})")
print("3. Save this model with the resized embeddings.")
print("4. **Crucially, you must then further pre-train or fine-tune this model on your Amharic dataset.**")
print("   Without this training, the model will not understand or effectively use the newly added Amharic tokens.")
