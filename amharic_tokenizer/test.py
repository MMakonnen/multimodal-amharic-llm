from transformers import AutoTokenizer

EXTENDED_TOKENIZER_SAVE_PATH = "./llama3_tokenizer_amharic_extended"

print("\nExample: Loading the extended tokenizer")
try:
    loaded_extended_tokenizer = AutoTokenizer.from_pretrained(EXTENDED_TOKENIZER_SAVE_PATH)
    print(f"Successfully loaded extended tokenizer. Vocab size: {loaded_extended_tokenizer.vocab_size}")
    amharic_text = "ይህ የአማርኛ የሙከራ ዓረፍተ ነገር ነው።" # "This is an Amharic test sentence."
    tokens = loaded_extended_tokenizer.tokenize(amharic_text)
    print(f"Tokenizing Amharic text '{amharic_text}': {tokens}")
    encoded_ids = loaded_extended_tokenizer.encode(amharic_text)
    print(f"Encoded IDs: {encoded_ids}")
except Exception as e:
    print(f"Error loading extended tokenizer for example: {e}")