from unsloth import FastLanguageModel
import torch
import math
import json
from datasets import Dataset


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="trainer_output/checkpoint-44000",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda:0",
)

# Load JSON file with input-output pairs
with open("processed_amharic_data/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Define a format function to concatenate input and output
def format_example(example):
    return {"text": example["input"] + " " + example["output"]}

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
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# Step 4: Calculate perplexity
ppl = compute_perplexity(model, tokenizer, dataset)
print(f"Perplexity on bottom 0.1% of dataset: {ppl}")