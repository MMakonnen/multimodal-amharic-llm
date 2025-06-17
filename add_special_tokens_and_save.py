from transformers import AutoTokenizer

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("rasyosef/Llama-3.2-1B-Amharic")

# 2. Add special tokens and save
new_special_tokens = ["<|begin_of_text|>", "<|user|>", "<|assistant|>", "<|end_of_turn|>"]
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
tokenizer.save_pretrained("./rasyosef_tokenizer_w_special_tokens")

print(f"Tokenizer with {len(tokenizer)} tokens saved to './my_extended_tokenizer'")