from unsloth import FastLanguageModel
import torch
import math
import json
from datasets import Dataset


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="finetuned_models/amharic_instruction_finetune_lr5e-05/20250619-141808_ours", # Change!
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda:0",
)

# Load JSON file with input-output pairs
with open("processed_amharic_data/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    data = data[:1000]

# Define a format function to concatenate input and output
def format_example(example):
    return {"text": example["input"] + "\n<|reserved_special_token_42|>\n" + example["output"],
            "prompt": example["input"] + "\n<|reserved_special_token_42|>\n"}

# Pipe through the format function
formatted_data = [format_example(example) for example in data]

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(formatted_data)


FastLanguageModel.for_inference(model)

# Step 3: Tokenize and compute loss
def compute_perplexity(model, tokenizer, dataset, max_length=2048):
    total_loss = 0.0
    total_tokens = 0

    for example in dataset:
        input_text = example["text"]
        tokenized_input = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = tokenized_input.input_ids.to(model.device)
        
        prompt = example["prompt"]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        prompt_input_ids = tokenized_prompt.input_ids.to(model.device)

        # We only want to compute the loss for the choice text, not the prompt.
        input_ids_clone = input_ids.clone()
        input_ids_clone[0, :len(prompt_input_ids[0])] = -100  # Mask the prompt part

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids_clone)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    print(f"Total tokens: {total_tokens}, Total loss: {total_loss}")
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# Step 4: Calculate perplexity
ppl = compute_perplexity(model, tokenizer, dataset)
print(f"Perplexity: {ppl}")