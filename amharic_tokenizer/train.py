import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from tokenizers import SentencePieceBPETokenizer # For training the new Amharic vocab component

# https://github.com/shubhamdawande/Extending-Llama-3-Tokenizer-Hindi can serve as a reference 

# Step 1: Define Parameters
LLAMA3_MODEL_ID = "meta-llama/Llama-3.2-1B"  # REPLACE with your Llama 3 model
EXTENDED_TOKENIZER_SAVE_PATH = "./llama3_tokenizer_amharic_extended"
NEW_AMHARIC_VOCAB_SIZE = 10000 # Adjust based on dataset size and needs

# Step 2: Load Datasets and Concatenate, taking only the 'text' column
print(f"Loading Amharic datasets")
datasets = [
    load_dataset("l-jiao/amharic-news", split="train"),
    load_dataset("l-jiao/amharic-wikipedia", split="train"),
    load_dataset("l-jiao/amharic-commoncrawl", split="train")
]
datasets = [dataset.remove_columns([col for col in dataset.column_names if col != "text"]) for dataset in datasets]
dataset = concatenate_datasets(datasets)

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
        else: 
            print(f"Skipping item with unexpected structure: {item}")
    print(f"Text iterator will yield {count} text samples.")

# Step 3: Train a New SentencePiece Tokenizer on Amharic Data
# This helps identify Amharic-specific subwords.
print(f"\nTraining a new SentencePiece BPE tokenizer for Amharic with vocab_size={NEW_AMHARIC_VOCAB_SIZE}...")
amharic_sp_tokenizer = SentencePieceBPETokenizer() 

# Train the SentencePiece model
amharic_sp_tokenizer.train_from_iterator(
    get_text_iterator(dataset),
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
